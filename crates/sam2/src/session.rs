// Modified work: ported from Hugging Face `transformers` `models/sam2_video/modeling_sam2_video.py`
// (Sam2VideoModel and Sam2VideoInferenceSession: memory-conditioned single-frame inference,
// memory bank selection, object pointers, propagation; Apache-2.0, Copyright The HuggingFace
// Inc. team) to Rust on candle. See NOTICE.
//! The video tracker: per-object memory banks, prompts on any frame, propagation forwards and
//! backwards with streaming results and cancellation.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Tensor};
use candle_nn::{Linear, ops::sigmoid};

use crate::decoder::FeedForward;
use crate::gpu::Gpu;
use crate::gpu::encoder::GpuEncoder;
use crate::gpu::memattn::GpuMemoryAttention;
use crate::gpu::memenc::GpuMemoryEncoder;
use crate::memattn::MemoryAttention;
use crate::memory::{Memory, MemoryEncoder};
use crate::model::{Encoded, ImageModel};
use crate::neck::ImageFeatures;
use crate::nn::{apply, linear};
use crate::prompt::{INPUT_SIZE, Prompt};
use crate::resize::antialias_bilinear;
use crate::weights::Weights;
use crate::{Error, Result};

/// Mask logit used where the object is absent (the reference's `NO_OBJ_SCORE`).
const NO_OBJECT: f32 = -1024.0;
/// Frames of memory the attention looks back over.
const MEMORY_FRAMES: usize = 7;
/// Object pointers kept per query.
const MAX_POINTERS: usize = 16;
/// Largest clip the session accepts.
const MAX_FRAMES: usize = 1_000_000;
const GRID: usize = 64;

/// Frames of the clip, already resized to the model input and normalised: `[1, 3, 1024, 1024]`.
pub trait FrameSource {
    fn num_frames(&self) -> usize;
    fn frame(&self, index: usize) -> Result<Tensor>;
}

/// Cooperative cancellation shared with the caller.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// One object's mask on one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameMask {
    pub frame: usize,
    pub object: ObjectId,
    /// Mask logits at 256 x 256 (positive is inside the object); `-1024` everywhere when the
    /// object is absent.
    pub logits: Vec<f32>,
    /// Selected decoder scores before optional video-predictor processing. Present only when a
    /// score processor is installed; shared when a conditioning result is retained or returned.
    pub decoder_logits: Option<Arc<Vec<f32>>>,
    /// The model's own estimate of the mask quality (0..1).
    pub iou: f32,
    /// Object-presence logit; at or below zero the object is not in the frame.
    pub object_score: f32,
    pub occluded: bool,
}

/// Which device runs the heavy parts (the backbone, the neck and the memory attention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Candle on the CPU: the reference path.
    Cpu,
    /// wgpu compute on the GPU (any vendor); an error if there is none.
    Gpu,
    /// The GPU when there is a real one and it loads, else the CPU. The environment variable
    /// `EFFECTCRAFT_SAM2_DEVICE` (`cpu`, `gpu` or `auto`) overrides the choice.
    Auto,
}

/// The GPU path: kernels for the backbone and memory attention. Device failures disable it;
/// memory-attention budget limits fall back to the CPU for that call only.
struct GpuParts {
    encoder: GpuEncoder,
    attention: GpuMemoryAttention,
    memory_encoder: GpuMemoryEncoder,
    ok: AtomicBool,
    adapter: String,
}

/// The weights of everything the video tracker needs.
pub struct VideoModel {
    gpu: Option<GpuParts>,
    pub image: ImageModel,
    memory_encoder: MemoryEncoder,
    memory_attention: MemoryAttention,
    temporal_pos: Vec<f32>,
    no_object_pointer: Tensor,
    pointer_proj: FeedForward,
    pointer_time_proj: Linear,
    /// `mask_downsample`: a 4 x 4 stride-4 convolution taking a mask prompt to the prompt encoder's resolution.
    mask_down: (Tensor, Tensor),
}

impl VideoModel {
    /// On the CPU: the reference path every test compares against.
    pub fn load(w: &Weights) -> Result<VideoModel> {
        VideoModel::load_with(w, Backend::Cpu)
    }

    /// On the GPU when there is one (see [`Backend::Auto`]).
    pub fn load_auto(w: &Weights) -> Result<VideoModel> {
        VideoModel::load_with(w, Backend::Auto)
    }

    pub fn load_with(w: &Weights, backend: Backend) -> Result<VideoModel> {
        let backend = match (backend, std::env::var("EFFECTCRAFT_SAM2_DEVICE").ok().as_deref()) {
            (Backend::Auto, Some("cpu")) => Backend::Cpu,
            (Backend::Auto, Some("gpu")) => Backend::Gpu,
            (b, _) => b,
        };
        let image = ImageModel::load(w)?;
        let gpu = match backend {
            Backend::Cpu => None,
            Backend::Gpu => Some(Self::load_gpu(w, &image)?),
            Backend::Auto => match Self::load_gpu(w, &image) {
                Ok(g) => Some(g),
                Err(e) => {
                    log::info!("SAM 2 runs on the CPU: {e}");
                    None
                }
            },
        };
        Ok(VideoModel {
            gpu,
            image,
            memory_encoder: MemoryEncoder::load(w)?,
            memory_attention: MemoryAttention::load(w)?,
            temporal_pos: w.get("memory_temporal_positional_encoding", &[MEMORY_FRAMES, 1, 1, 64])?.flatten_all()?.to_vec1::<f32>()?,
            no_object_pointer: w.get("no_object_pointer", &[1, 256])?,
            pointer_proj: FeedForward::load(w, "object_pointer_proj", 256, 256, 256, 3, false)?,
            pointer_time_proj: linear(w, "temporal_positional_encoding_projection_layer", 256, 64)?,
            mask_down: (w.get("mask_downsample.weight", &[1, 1, 4, 4])?, w.get("mask_downsample.bias", &[1])?),
        })
    }

    fn load_gpu(w: &Weights, image: &ImageModel) -> Result<GpuParts> {
        let gpu = Gpu::shared()?;
        Ok(GpuParts {
            encoder: GpuEncoder::load(gpu.clone(), w, image.vision.hiera.clone())?,
            attention: GpuMemoryAttention::load(gpu.clone(), w)?,
            memory_encoder: GpuMemoryEncoder::load(gpu.clone(), w)?,
            ok: AtomicBool::new(true),
            adapter: gpu.adapter_name().to_string(),
        })
    }

    /// The device the heavy parts run on now: the GPU's name, or "CPU".
    pub fn device_name(&self) -> String {
        match &self.gpu {
            Some(g) if g.ok.load(Ordering::Relaxed) => g.adapter.clone(),
            _ => "CPU".to_string(),
        }
    }

    /// Run on the CPU from now on (a GPU that misbehaves, or the user's choice). Weights and results are
    /// unchanged; only the device differs.
    pub fn disable_gpu(&self) {
        if let Some(g) = &self.gpu {
            g.ok.store(false, Ordering::Relaxed);
        }
    }

    fn gpu_failed(&self, what: &str, e: &Error) {
        if let Some(g) = &self.gpu
            && g.ok.swap(false, Ordering::Relaxed)
        {
            log::warn!("SAM 2 {what} failed on the GPU ({e}); the CPU takes over");
        }
    }

    /// Encode a normalised `[1, 3, 1024, 1024]` frame on the GPU if it is working, else on the CPU.
    pub fn encode(&self, pixels: &Tensor) -> Result<Encoded> {
        if let Some(g) = self.gpu.as_ref().filter(|g| g.ok.load(Ordering::Relaxed)) {
            match g.encoder.encode(pixels) {
                Ok(f) => {
                    let none = || Tensor::zeros((1, 0, 0, 0), DType::F32, &candle_core::Device::Cpu);
                    let pos = self.image.vision.neck.position(GRID, GRID)?;
                    let features = ImageFeatures { features: vec![none()?, none()?, f.low], positions: vec![none()?, none()?, pos] };
                    return Ok(Encoded { features, high: f.high });
                }
                Err(e) => self.gpu_failed("encoding", &e),
            }
        }
        self.image.encode(pixels)
    }

    /// The memory of a frame (its features and the scaled mask), on the GPU if it is working, else on the CPU.
    fn memorize(&self, features: &Tensor, mask: &Tensor, present: bool) -> Result<Memory> {
        if let Some(g) = self.gpu.as_ref().filter(|g| g.ok.load(Ordering::Relaxed)) {
            match g.memory_encoder.encode(features, mask).and_then(|t| self.memory_encoder.finish(&t, present)) {
                Ok(m) => return Ok(m),
                Err(e) => self.gpu_failed("memory encoding", &e),
            }
        }
        self.memory_encoder.memory(features, mask, present)
    }

    /// Memory attention on the GPU if it is working, else on the CPU.
    fn attend(&self, current: &Tensor, current_pos: &Tensor, memory: &Tensor, memory_pos: &Tensor, pointers: usize) -> Result<Tensor> {
        if let Some(g) = self.gpu.as_ref().filter(|g| g.ok.load(Ordering::Relaxed)) {
            match g.attention.forward(current, current_pos, memory, memory_pos, pointers) {
                Ok(t) => return Ok(t),
                Err(Error::Budget(reason)) => log::info!("SAM 2 memory attention on the CPU for this call: {reason}"),
                Err(e) => self.gpu_failed("memory attention", &e),
            }
        }
        self.memory_attention.forward(current, current_pos, memory, memory_pos, pointers)
    }

    pub fn open(path: &Path) -> Result<VideoModel> {
        VideoModel::load(&Weights::open(path)?)
    }
}

/// What the tracker keeps for a frame it has processed.
#[derive(Clone)]
struct FrameOut {
    memory: Arc<Memory>,
    /// `[1, 256]`
    pointer: Tensor,
    /// The mask of a conditioning frame (returned again, unchanged, when propagation passes it).
    /// [`CorrectionPolicy::Official`] also keeps it on non-conditioning frames: its processed
    /// scores are the previous-mask input of a later correction click. Shared, never copied.
    result: Option<Arc<FrameMask>>,
}

/// How the tracker treats a prompt on a frame propagation has already traversed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CorrectionPolicy {
    /// A prompt on a frame is a correction only if the frame is a retained non-conditioning
    /// memory; prompts are encoded into memory at once; propagation prunes distant memories.
    #[default]
    Legacy,
    /// Meta's SAM 2.1 video predictor (`sam2_video_predictor.py`, rev `2b90b9f`): a frame is an
    /// initial conditioning frame until propagation has traversed it (then a correction that is
    /// decoded with the consolidated memory bank in the direction it was traversed). Prompt outputs
    /// stay pending, without memory, until the next propagation consolidates them; nothing is
    /// pruned. Pending output is staged separately and committed only whole.
    Official,
}

/// An interactive output not yet consolidated into the memory bank ([`CorrectionPolicy::Official`]).
#[derive(Clone)]
struct Pending {
    /// An initial conditioning frame (decoded without memory); else a correction of a traversed frame.
    conditioning: bool,
    pointer: Tensor,
    /// The `[1, 1, 1024, 1024]` mask the memory encoder takes (already scaled to +-10).
    for_memory: Tensor,
    present: bool,
    result: Arc<FrameMask>,
}

#[derive(Default, Clone)]
struct ObjectState {
    cond: BTreeMap<usize, FrameOut>,
    non_cond: BTreeMap<usize, FrameOut>,
    /// [`CorrectionPolicy::Official`]: frame -> direction (`reverse`) propagation last traversed it.
    /// Set only when an accepted traversal publishes the frame; independent of memory membership.
    tracked: BTreeMap<usize, bool>,
    pending: BTreeMap<usize, Pending>,
}

/// What decoding one frame produced, before its memory is made.
struct Decoded {
    feats: Arc<FrameFeatures>,
    /// `[1, 1, 1024, 1024]` memory-encoder mask input.
    for_memory: Tensor,
    present: bool,
    pointer: Tensor,
    result: FrameMask,
}

struct FrameFeatures {
    /// Lowest-resolution level `[1, 256, 64, 64]` and its position encoding.
    low: Tensor,
    pos: Tensor,
    high: (Tensor, Tensor),
}

/// What a tracker may hold and take. A call that would exceed it stops with [`Error::Budget`].
/// Under [`CorrectionPolicy::Official`] `max_state_bytes` bounds the unique backing payloads (memory
/// tensors, pointers, processed and raw scores, pending memory input, cached features) that
/// accepted state, the cache and live prepared transitions hold at once, old and replacement
/// together. It excludes model weights, scratch, allocator overhead and host copies; it is not a
/// bound on total heap or GPU memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    /// Memory banks, results and cached frame features (bytes).
    pub max_state_bytes: usize,
    /// Wall-clock seconds one `propagate` call may take (`None`: no limit).
    pub max_seconds: Option<f64>,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget { max_state_bytes: 2 << 30, max_seconds: None }
    }
}

/// Bytes of one stored frame: memory features and position encoding `[4096, 64]`, the pointer.
const FRAME_BYTES: usize = 2 * GRID * GRID * 64 * 4 + 256 * 4;
/// Bytes of a conditioning frame's kept result (`256 x 256` logits).
const RESULT_BYTES: usize = 4 * GRID * 4 * GRID * 4;
/// Bytes of one frame's cached neck features (lowest level and position, two skip levels).
const FEATURE_BYTES: usize = (2 * 256 * GRID * GRID + 32 * 16 * GRID * GRID + 64 * 4 * GRID * GRID) * 4;

pub struct VideoTracker<'a> {
    model: Arc<VideoModel>,
    frames: Box<dyn FrameSource + 'a>,
    objects: BTreeMap<ObjectId, ObjectState>,
    cache: VecDeque<(usize, Arc<FrameFeatures>)>,
    budget: Budget,
    peak: usize,
    score_processor: Option<Box<dyn FnMut(&[f32]) -> Result<Vec<f32>> + 'a>>,
    correction: CorrectionPolicy,
    /// Official policy: payload bytes reserved by the live prepared transition.
    reserved: usize,
    /// Official policy: feature sets the live preparation still references (cache eviction does
    /// not release them).
    held: Vec<Arc<FrameFeatures>>,
    preparing: bool,
    /// Start of the host operation (propagation) whose deadline covers its transitions.
    op_started: Option<std::time::Instant>,
}

/// What a prepared transition asks of the tracker ([`VideoTracker::prepare`]). Inputs are borrowed
/// for the preparation only. Every kind but `Reset` needs [`CorrectionPolicy::Official`].
pub enum PrepareRequest<'p> {
    /// A click/box prompt; `mask_logits` is the previous-mask input (`[1, 1, 256, 256]` F32,
    /// finite, clamped to [-32, 32] here, where it is consumed).
    Prompt { frame: usize, object: ObjectId, prompt: &'p Prompt, mask_logits: Option<&'p Tensor> },
    /// A dense mask prompt: `[1, 1, 1024, 1024]` of 0 and 1.
    Dense { frame: usize, object: ObjectId, mask: &'p Tensor },
    /// Consolidate every pending output into the memory banks.
    Preflight,
    /// Produce one frame of one object (a cached conditioning output is only marked traversed).
    Advance { frame: usize, object: ObjectId, direction: Direction },
    /// Forget every object and cached frame (policy and processor are kept).
    Reset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionKind {
    Prompt,
    Dense,
    Preflight,
    Advance,
    Reset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionInfo {
    pub kind: TransitionKind,
    pub frame: Option<usize>,
    pub object: Option<ObjectId>,
    pub direction: Option<Direction>,
}

#[derive(Clone, Copy)]
enum Slot {
    Pending,
    Cond,
    NonCond,
}

/// The complete next semantic state, built before anything is published.
struct Stage {
    info: TransitionInfo,
    next: BTreeMap<ObjectId, ObjectState>,
    result: Option<(ObjectId, usize, Slot)>,
    clear_cache: bool,
}

impl Stage {
    fn result(&self) -> Option<&FrameMask> {
        let (object, frame, slot) = self.result.as_ref()?;
        let o = self.next.get(object)?;
        match slot {
            Slot::Pending => o.pending.get(frame).map(|p| &*p.result),
            Slot::Cond => o.cond.get(frame).and_then(|f| f.result.as_deref()),
            Slot::NonCond => o.non_cond.get(frame).and_then(|f| f.result.as_deref()),
        }
    }
}

/// A prepared, not yet accepted transition. It exclusively borrows the tracker, so nothing else
/// (reset, policy or processor change, another preparation) can intervene. Dropping it aborts.
/// No semantic state has changed; only the disposable feature cache may have.
pub struct PreparedTransition<'t, 'a> {
    tracker: Option<&'t mut VideoTracker<'a>>,
    stage: Option<Stage>,
    started: std::time::Instant,
}

/// A validated transition: [`ReadyTransition::commit`] publishes it and cannot fail. Dropping it aborts.
pub struct ReadyTransition<'t, 'a> {
    tracker: Option<&'t mut VideoTracker<'a>>,
    stage: Option<Stage>,
}

impl<'t, 'a> PreparedTransition<'t, 'a> {
    pub fn info(&self) -> TransitionInfo {
        self.stage.as_ref().map_or(TransitionInfo { kind: TransitionKind::Reset, frame: None, object: None, direction: None }, |s| s.info)
    }

    /// The mask this transition produces (prompt, dense, advance); `None` for preflight and reset.
    pub fn result(&self) -> Option<&FrameMask> {
        self.stage.as_ref().and_then(Stage::result)
    }

    /// The final cancellation, deadline and invariant checks. Cancellation racing after this
    /// returns applies to the next transition.
    pub fn validate(mut self, cancel: &Cancel) -> Result<ReadyTransition<'t, 'a>> {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        {
            let tracker = self.tracker.as_deref().ok_or_else(|| Error::Compute("transition already finished".into()))?;
            tracker.check_deadline(self.started)?;
            let stage = self.stage.as_ref().ok_or_else(|| Error::Compute("transition already finished".into()))?;
            if stage.result.is_some() && stage.result().is_none() {
                return Err(Error::Compute("prepared transition lost its result".into()));
            }
        }
        Ok(ReadyTransition { tracker: self.tracker.take(), stage: self.stage.take() })
    }

    pub fn abort(self) {}
}

impl Drop for PreparedTransition<'_, '_> {
    fn drop(&mut self) {
        if let Some(t) = self.tracker.take() {
            t.release();
        }
    }
}

impl<'t, 'a> ReadyTransition<'t, 'a> {
    pub fn info(&self) -> TransitionInfo {
        self.stage.as_ref().map_or(TransitionInfo { kind: TransitionKind::Reset, frame: None, object: None, direction: None }, |s| s.info)
    }

    pub fn result(&self) -> Option<&FrameMask> {
        self.stage.as_ref().and_then(Stage::result)
    }

    /// Publish: swap the prepared state in. No inference, callbacks, I/O, allocation or checks.
    pub fn commit(mut self) {
        if let (Some(t), Some(stage)) = (self.tracker.take(), self.stage.take()) {
            let old = std::mem::replace(&mut t.objects, stage.next);
            if stage.clear_cache {
                t.cache.clear();
            }
            t.release();
            drop(old);
        }
    }

    pub fn abort(self) {}
}

impl Drop for ReadyTransition<'_, '_> {
    fn drop(&mut self) {
        if let Some(t) = self.tracker.take() {
            t.release();
        }
    }
}

fn tensor_bytes(t: &Tensor) -> usize {
    t.elem_count().saturating_mul(t.dtype().size_in_bytes())
}

fn mask_bytes(m: &FrameMask) -> usize {
    m.logits.len().saturating_mul(4).saturating_add(m.decoder_logits.as_ref().map_or(0, |v| v.len().saturating_mul(4)))
}

fn out_bytes(o: &FrameOut) -> usize {
    tensor_bytes(&o.memory.features).saturating_add(tensor_bytes(&o.memory.pos)).saturating_add(tensor_bytes(&o.pointer)).saturating_add(o.result.as_deref().map_or(0, mask_bytes))
}

fn pending_bytes(p: &Pending) -> usize {
    tensor_bytes(&p.pointer).saturating_add(tensor_bytes(&p.for_memory)).saturating_add(mask_bytes(&p.result))
}

fn features_bytes(f: &FrameFeatures) -> usize {
    [&f.low, &f.pos, &f.high.0, &f.high.1].iter().fold(0usize, |a, t| a.saturating_add(tensor_bytes(t)))
}

/// A compact owned copy: a retained payload never keeps a larger parent allocation alive.
fn compact(t: &Tensor) -> Result<Tensor> {
    Ok(t.affine(1.0, 0.0)?)
}

fn sine_embedding_1d(positions: &[f32], dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let dim_t: Vec<f32> = (0..half).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / half as f32)).collect();
    let mut out = Vec::with_capacity(positions.len() * dim);
    for p in positions {
        out.extend(dim_t.iter().map(|d| (p / d).sin()));
        out.extend(dim_t.iter().map(|d| (p / d).cos()));
    }
    Ok(Tensor::from_vec(out, (positions.len(), dim), &candle_core::Device::Cpu)?)
}

fn scalar(t: &Tensor) -> Result<f32> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?.first().copied().unwrap_or(0.0))
}

impl<'a> VideoTracker<'a> {
    pub fn new(model: Arc<VideoModel>, frames: Box<dyn FrameSource + 'a>) -> Result<VideoTracker<'a>> {
        let n = frames.num_frames();
        if n == 0 || n > MAX_FRAMES {
            return Err(Error::Input(format!("a clip needs between 1 and {MAX_FRAMES} frames, got {n}")));
        }
        Ok(VideoTracker { model, frames, objects: BTreeMap::new(), cache: VecDeque::new(), budget: Budget::default(), peak: 0, score_processor: None, correction: CorrectionPolicy::Legacy, reserved: 0, held: Vec::new(), preparing: false, op_started: None })
    }

    pub fn num_frames(&self) -> usize {
        self.frames.num_frames()
    }

    /// Replace the limits (see [`Budget`]).
    pub fn set_budget(&mut self, budget: Budget) {
        self.budget = budget;
    }

    /// Install host-owned low-resolution video-predictor processing. Existing users keep the raw
    /// path. Change policy only on an empty tracker, so cached results never mix policies.
    /// The host may reuse its GPU connected-component implementation without adding a second GPU
    /// runtime here. Unaffected scores must be preserved by the processor.
    pub fn set_score_processor(&mut self, processor: Option<Box<dyn FnMut(&[f32]) -> Result<Vec<f32>> + 'a>>) -> Result<()> {
        if !self.objects.is_empty() {
            return Err(Error::Input("reset the tracker before changing video-predictor processing".into()));
        }
        self.score_processor = processor;
        Ok(())
    }

    /// Choose how corrections on traversed frames work (see [`CorrectionPolicy`]). Only on an empty
    /// tracker, so memories of two policies never mix.
    pub fn set_correction_policy(&mut self, policy: CorrectionPolicy) -> Result<()> {
        if !self.objects.is_empty() {
            return Err(Error::Input("reset the tracker before changing its correction policy".into()));
        }
        self.correction = policy;
        Ok(())
    }

    pub fn correction_policy(&self) -> CorrectionPolicy {
        self.correction
    }

    /// Apply the installed policy without changing tracker state. This also lets the host derive
    /// the next click's input from separately persisted decoder scores.
    pub fn process_mask_scores(&mut self, scores: &[f32]) -> Result<Vec<f32>> {
        if scores.len() != RESULT_BYTES / 4 || scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::Input("predictor scores must be finite 256x256 F32".into()));
        }
        let processed = match self.score_processor.as_mut() {
            Some(f) => f(scores)?,
            None => scores.to_vec(),
        };
        if processed.len() != scores.len() || processed.iter().any(|v| !v.is_finite()) {
            return Err(Error::Input("video-predictor processor returned malformed scores".into()));
        }
        Ok(processed)
    }

    /// Bytes charged now: stored frames, kept results and cached features (under
    /// [`CorrectionPolicy::Official`]: the unique payloads of [`Budget`], prepared ones included).
    pub fn state_bytes(&self) -> usize {
        if self.correction == CorrectionPolicy::Official {
            return self.live_bytes();
        }
        let frames: usize = self.objects.values().map(|o| (o.cond.len() + o.non_cond.len()) * FRAME_BYTES + o.cond.len() * RESULT_BYTES).sum();
        let raw: usize = self.objects.values().flat_map(|o| o.cond.values()).filter_map(|o| o.result.as_ref()).filter_map(|r| r.decoder_logits.as_ref()).map(|v| v.len() * 4).sum();
        frames + raw + self.cache.len() * FEATURE_BYTES
    }

    /// Unique retained payload bytes: accepted state, the feature cache, features a preparation
    /// still references, and the payloads a prepared transition reserved.
    fn live_bytes(&self) -> usize {
        let mut total = 0usize;
        for o in self.objects.values() {
            for f in o.cond.values().chain(o.non_cond.values()) {
                total = total.saturating_add(out_bytes(f));
            }
            for p in o.pending.values() {
                total = total.saturating_add(pending_bytes(p));
            }
        }
        let mut seen: Vec<*const FrameFeatures> = Vec::new();
        for f in self.cache.iter().map(|(_, f)| f).chain(self.held.iter()) {
            let ptr = Arc::as_ptr(f);
            if !seen.contains(&ptr) {
                seen.push(ptr);
                total = total.saturating_add(features_bytes(f));
            }
        }
        total.saturating_add(self.reserved)
    }

    /// Reserve `bytes` of retained payload for the live prepared transition.
    fn admit(&mut self, bytes: usize) -> Result<()> {
        let used = self.live_bytes();
        let total = used.checked_add(bytes).ok_or_else(|| Error::Budget("the tracker's retained bytes overflow".into()))?;
        if total > self.budget.max_state_bytes {
            return Err(Error::Budget(format!("the tracker holds {used} bytes and this step needs {bytes} more, over its {} byte limit", self.budget.max_state_bytes)));
        }
        self.reserved = self.reserved.saturating_add(bytes);
        self.peak = self.peak.max(total);
        Ok(())
    }

    fn check_live(&mut self) -> Result<()> {
        let used = self.live_bytes();
        if used > self.budget.max_state_bytes {
            return Err(Error::Budget(format!("the tracker holds {used} bytes, over its {} byte limit", self.budget.max_state_bytes)));
        }
        self.peak = self.peak.max(used);
        Ok(())
    }

    fn check_deadline(&self, started: std::time::Instant) -> Result<()> {
        if let Some(limit) = self.budget.max_seconds.filter(|l| started.elapsed().as_secs_f64() > *l) {
            return Err(Error::Budget(format!("tracking took more than {limit} s")));
        }
        Ok(())
    }

    fn release(&mut self) {
        self.reserved = 0;
        self.held.clear();
        self.preparing = false;
    }

    /// The most [`VideoTracker::state_bytes`] reached so far.
    pub fn peak_state_bytes(&self) -> usize {
        self.peak
    }

    fn check_budget(&mut self) -> Result<()> {
        let used = self.state_bytes();
        self.peak = self.peak.max(used);
        if used > self.budget.max_state_bytes {
            return Err(Error::Budget(format!("the tracker holds {} MB, over its {} MB limit", used >> 20, self.budget.max_state_bytes >> 20)));
        }
        Ok(())
    }

    /// Forget every object and cached frame (the clip stays).
    pub fn reset(&mut self) {
        self.objects.clear();
        self.cache.clear();
    }

    fn features(&mut self, frame: usize) -> Result<Arc<FrameFeatures>> {
        let f = self.cached_features(frame)?;
        if self.preparing {
            self.held.push(f.clone());
            self.check_live()?;
        }
        Ok(f)
    }

    fn cached_features(&mut self, frame: usize) -> Result<Arc<FrameFeatures>> {
        if let Some((_, f)) = self.cache.iter().find(|(i, _)| *i == frame) {
            return Ok(f.clone());
        }
        let pixels = self.frames.frame(frame)?;
        let enc = self.model.encode(&pixels)?;
        let (low, pos) = match (enc.features.features.get(2), enc.features.positions.get(2)) {
            (Some(l), Some(p)) => (l.clone(), p.clone()),
            _ => return Err(Error::Compute("vision encoder returned too few levels".into())),
        };
        let f = Arc::new(FrameFeatures { low, pos, high: enc.high });
        self.cache.push_back((frame, f.clone()));
        while self.cache.len() > 2 {
            self.cache.pop_front();
        }
        Ok(f)
    }

    /// The memory bank for `frame`: memory frames and object pointers with their position
    /// encodings, plus how many trailing tokens are pointers.
    fn memory_bank(&self, obj: &ObjectState, frame: usize, reverse: bool) -> Result<(Tensor, Tensor, usize)> {
        let (mut feats, mut poss) = (vec![], vec![]);
        let temporal = |off: usize| -> Result<Tensor> {
            // The reference indexes with `offset - 1`, so the conditioning frames (offset 0) use the last row.
            let row = if off == 0 { MEMORY_FRAMES - 1 } else { off - 1 };
            let v = self.model.temporal_pos.get(row * 64..(row + 1) * 64).ok_or_else(|| Error::Weights("temporal position table".into()))?;
            Ok(Tensor::from_slice(v, (1, 64), &candle_core::Device::Cpu)?)
        };
        for out in obj.cond.values() {
            feats.push(out.memory.features.clone());
            poss.push(out.memory.pos.broadcast_add(&temporal(0)?)?);
        }
        for off in (1..MEMORY_FRAMES).rev() {
            let prev = if reverse { frame.checked_add(off) } else { frame.checked_sub(off) };
            if let Some(out) = prev.and_then(|p| obj.non_cond.get(&p)) {
                feats.push(out.memory.features.clone());
                poss.push(out.memory.pos.broadcast_add(&temporal(off)?)?);
            }
        }
        // Object pointers: conditioning frames on the already-tracked side, then recent frames.
        let n_frames = self.frames.num_frames();
        let max_ptrs = n_frames.min(MAX_POINTERS);
        let (mut offsets, mut pointers): (Vec<f32>, Vec<Tensor>) = (vec![], vec![]);
        let sign = if reverse { -1.0f32 } else { 1.0 };
        for (t, out) in &obj.cond {
            if if reverse { *t >= frame } else { *t <= frame } {
                offsets.push((frame as f32 - *t as f32) * sign);
                pointers.push(out.pointer.clone());
            }
        }
        for off in 1..max_ptrs {
            let r = if reverse { frame.checked_add(off) } else { frame.checked_sub(off) };
            let Some(r) = r.filter(|r| *r < n_frames) else { break };
            if let Some(out) = obj.non_cond.get(&r) {
                offsets.push(off as f32);
                pointers.push(out.pointer.clone());
            }
        }
        let mut n_pointer_tokens = 0;
        if !pointers.is_empty() {
            let max_diff = (max_ptrs.saturating_sub(1)).max(1) as f32;
            let norm: Vec<f32> = offsets.iter().map(|o| o / max_diff).collect();
            let pe = apply(&self.model.pointer_time_proj, &sine_embedding_1d(&norm, 256)?)?;
            let ptrs = Tensor::cat(&pointers, 0)?; // [P, 256]
            let p = ptrs.dim(0)?;
            let tokens = ptrs.reshape((p * 4, 64))?;
            let pe = pe.unsqueeze(1)?.broadcast_as((p, 4, 64))?.reshape((p * 4, 64))?;
            n_pointer_tokens = p * 4;
            feats.push(tokens);
            poss.push(pe.contiguous()?);
        }
        if feats.is_empty() {
            return Err(Error::Input("the object has no conditioning frame; add a prompt first".into()));
        }
        Ok((Tensor::cat(&feats, 0)?, Tensor::cat(&poss, 0)?, n_pointer_tokens))
    }

    /// Run the model on one frame for one object and store the result.
    fn step(&mut self, id: ObjectId, frame: usize, prompt: Option<&Prompt>, reverse: bool, conditioning: bool) -> Result<FrameMask> {
        self.step_with_mask(id, frame, prompt, None, reverse, conditioning)
    }

    /// [`Self::step`] with an optional previous-mask dense prompt (`[1, 1, 256, 256]` logits,
    /// already validated and clamped to [-32, 32] by the caller).
    fn step_with_mask(&mut self, id: ObjectId, frame: usize, prompt: Option<&Prompt>, mask_logits: Option<&Tensor>, reverse: bool, conditioning: bool) -> Result<FrameMask> {
        let profile = std::env::var_os("EFFECTCRAFT_SAM2_PROFILE").is_some();
        let d = self.decode_frame(id, frame, prompt, mask_logits, reverse, !conditioning)?;
        let lap = std::time::Instant::now();
        let memory = self.model.memorize(&d.feats.low, &d.for_memory, d.present)?;
        if profile {
            eprintln!("sam2 frame {frame}: memory encoder {:.0} ms", lap.elapsed().as_secs_f64() * 1e3);
        }
        let Decoded { pointer, result, .. } = d;
        let state = self.objects.entry(id).or_default();
        let entry = FrameOut { memory: Arc::new(memory), pointer, result: conditioning.then(|| Arc::new(result.clone())) };
        if conditioning {
            state.non_cond.remove(&frame);
            state.cond.insert(frame, entry);
        } else {
            state.non_cond.insert(frame, entry);
            Self::prune(state, frame);
        }
        self.check_budget()?;
        Ok(result)
    }

    /// Run the model on `frame` up to the memory input: attend to the memory bank when
    /// `use_memory` (else the no-memory embedding), decode, apply the video-predictor score
    /// processing and make the object pointer. Changes no tracker state.
    fn decode_frame(&mut self, id: ObjectId, frame: usize, prompt: Option<&Prompt>, mask_logits: Option<&Tensor>, reverse: bool, use_memory: bool) -> Result<Decoded> {
        // EFFECTCRAFT_SAM2_PROFILE=1 prints where a frame's time goes (stderr).
        let profile = std::env::var_os("EFFECTCRAFT_SAM2_PROFILE").is_some();
        let mut lap = std::time::Instant::now();
        let mut mark = |what: &str| {
            if profile {
                eprintln!("sam2 frame {frame}: {what} {:.0} ms", lap.elapsed().as_secs_f64() * 1e3);
            }
            lap = std::time::Instant::now();
        };
        let feats = self.features(frame)?;
        mark("encode");
        let model = self.model.clone();
        let tokens = |t: &Tensor| -> Result<Tensor> { Ok(t.flatten_from(2)?.squeeze(0)?.t()?.contiguous()?) };
        let conditioned = if !use_memory {
            feats.low.broadcast_add(model.image.no_memory())?
        } else {
            let obj = self.objects.get(&id).ok_or_else(|| Error::Input("unknown object".into()))?;
            let (mem, mem_pos, n_ptr) = self.memory_bank(obj, frame, reverse)?;
            let out = model.attend(&tokens(&feats.low)?, &tokens(&feats.pos)?, &mem, &mem_pos, n_ptr)?;
            out.t()?.contiguous()?.reshape((1, 256, GRID, GRID))?
        };
        mark("memory attention");
        let num_points = prompt.map_or(0, |p| p.points.len() + 2 * usize::from(p.bbox.is_some()));
        let multimask = num_points <= 1;
        let sparse = match prompt {
            Some(p) => model.image.prompt.sparse(p)?,
            None => model.image.prompt.sparse_none()?,
        };
        let dense = match mask_logits {
            Some(m) => model.image.prompt.dense_mask(m)?,
            None => model.image.prompt.dense_empty()?,
        };
        let out = model.image.decode_embedded(&conditioned, &feats.high, &sparse, &dense, multimask)?;

        mark("decoder");
        let score = scalar(&out.object_score)?;
        let present = score > 0.0;
        let ious = out.iou.flatten_all()?.to_vec1::<f32>()?;
        let best = if multimask { ious.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0 } else { 0 };
        let low = out.masks.narrow(1, best, 1)?;
        let low = if present { low } else { Tensor::full(NO_OBJECT, low.dims(), low.device())? };
        let token = if multimask { out.sam_tokens.narrow(1, best, 1)? } else { out.sam_tokens.narrow(1, 0, 1)? }.squeeze(1)?;
        let pointer = if present { model.pointer_proj.forward(&token)? } else { model.no_object_pointer.clone() };

        let decoder_logits = self.score_processor.as_ref().map(|_| low.flatten_all()?.to_vec1::<f32>().map(Arc::new)).transpose()?;
        let processed = if let Some(raw) = decoder_logits.as_ref() {
            Some(Tensor::from_vec(self.process_mask_scores(raw)?, low.shape(), low.device())?)
        } else {
            None
        };
        // Interactive predictions are processed before their memory is made. Propagated memory
        // uses the decoder prediction, while its returned/cached output is processed afterwards.
        let memory_low = if prompt.is_some() { processed.as_ref().unwrap_or(&low) } else { &low };
        let high = memory_low.upsample_bilinear2d(INPUT_SIZE, INPUT_SIZE, false)?;
        let for_memory = if prompt.is_some() { high.gt(0f32)?.to_dtype(DType::F32)? } else { sigmoid(&high)? };
        let for_memory = ((for_memory * 20.0)? - 10.0)?;
        mark("upsample and mask");

        let iou = ious.get(best).copied().unwrap_or(0.0);
        let result = FrameMask { frame, object: id, logits: processed.as_ref().unwrap_or(&low).flatten_all()?.to_vec1::<f32>()?, decoder_logits, iou, object_score: score, occluded: !present };
        Ok(Decoded { feats, for_memory, present, pointer, result })
    }

    // ───────────────────────── CorrectionPolicy::Official ─────────────────────────

    /// The previous-mask input of the next correction click on `frame`, as Meta looks it up: the
    /// accepted pending output of the same kind (initial or correction), else the consolidated
    /// conditioning output, else the consolidated non-conditioning output. Policy-processed scores,
    /// unclamped (the consumer clamps them to [-32, 32]). `None` before any accepted output exists
    /// for the frame; an unaccepted candidate is never exposed.
    pub fn latest_feedback(&self, object: ObjectId, frame: usize) -> Option<&[f32]> {
        let o = self.objects.get(&object)?;
        let initial = !o.tracked.contains_key(&frame);
        if let Some(p) = o.pending.get(&frame).filter(|p| p.conditioning == initial) {
            return Some(p.result.logits.as_slice());
        }
        o.cond.get(&frame).and_then(|f| f.result.as_deref()).or_else(|| o.non_cond.get(&frame).and_then(|f| f.result.as_deref())).map(|r| r.logits.as_slice())
    }

    /// [`Self::latest_feedback`], copied.
    pub fn previous_scores(&self, object: ObjectId, frame: usize) -> Option<Vec<f32>> {
        self.latest_feedback(object, frame).map(<[f32]>::to_vec)
    }

    /// Whether an interactive output awaits consolidation (see [`Self::commit_pending`]).
    pub fn has_pending(&self) -> bool {
        self.objects.values().any(|o| !o.pending.is_empty())
    }

    /// Whether propagation has traversed `frame` for `object`, and in which direction (`true` = backward).
    pub fn traversed(&self, object: ObjectId, frame: usize) -> Option<bool> {
        self.objects.get(&object)?.tracked.get(&frame).copied()
    }

    /// Prepare one transition of the [`CorrectionPolicy::Official`] tracker without changing its
    /// semantic state (the disposable feature cache may change). All model work, processing,
    /// allocation and budget reservation happen here; `validate` then `commit` publish it. The
    /// returned handle borrows the tracker exclusively until committed, aborted or dropped.
    pub fn prepare<'t>(&'t mut self, request: PrepareRequest<'_>, cancel: &Cancel) -> Result<PreparedTransition<'t, 'a>> {
        let reset = matches!(request, PrepareRequest::Reset);
        let started = match self.begin(cancel, reset) {
            Ok(s) => s,
            Err(e) => {
                self.release();
                return Err(e);
            }
        };
        match self.build(request, cancel, started) {
            Ok(stage) => Ok(PreparedTransition { tracker: Some(self), stage: Some(stage), started }),
            Err(e) => {
                self.release();
                Err(e)
            }
        }
    }

    fn begin(&mut self, cancel: &Cancel, reset: bool) -> Result<std::time::Instant> {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        self.reserved = 0;
        self.held.clear();
        self.preparing = true;
        if !reset {
            if self.budget.max_seconds.is_some_and(|s| !s.is_finite() || s < 0.0) {
                return Err(Error::Input("max_seconds must be finite and not negative".into()));
            }
            if self.live_bytes() > self.budget.max_state_bytes {
                // The cache is disposable: evict it first. Accepted state still over the limit rejects.
                self.cache.clear();
            }
            let used = self.live_bytes();
            if used > self.budget.max_state_bytes {
                return Err(Error::Budget(format!("the tracker holds {used} bytes, over its {} byte limit; reset it", self.budget.max_state_bytes)));
            }
        }
        Ok(self.op_started.unwrap_or_else(std::time::Instant::now))
    }

    fn checkpoint(&self, cancel: &Cancel, started: std::time::Instant) -> Result<()> {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        self.check_deadline(started)
    }

    fn build(&mut self, request: PrepareRequest<'_>, cancel: &Cancel, started: std::time::Instant) -> Result<Stage> {
        if let PrepareRequest::Reset = request {
            return Ok(Stage { info: TransitionInfo { kind: TransitionKind::Reset, frame: None, object: None, direction: None }, next: BTreeMap::new(), result: None, clear_cache: true });
        }
        if self.correction != CorrectionPolicy::Official {
            return Err(Error::Input("prepared transitions need the official correction policy".into()));
        }
        match request {
            PrepareRequest::Prompt { frame, object, prompt, mask_logits } => self.build_prompt(frame, object, prompt, mask_logits, cancel, started),
            PrepareRequest::Dense { frame, object, mask } => self.build_dense(frame, object, mask, cancel, started),
            PrepareRequest::Preflight => self.build_preflight(cancel, started),
            PrepareRequest::Advance { frame, object, direction } => self.build_advance(frame, object, direction, cancel, started),
            PrepareRequest::Reset => Err(Error::Compute("unreachable reset request".into())),
        }
    }

    /// Meta's `add_new_points_or_box`: a frame propagation has not traversed is an initial
    /// conditioning frame (no memory); a traversed one is a correction decoded with the consolidated
    /// memory bank (frozen while outputs are pending), in the direction it was traversed.
    fn build_prompt(&mut self, frame: usize, object: ObjectId, prompt: &Prompt, mask_logits: Option<&Tensor>, cancel: &Cancel, started: std::time::Instant) -> Result<Stage> {
        self.check_frame(frame)?;
        prompt.validate()?;
        if prompt.points.is_empty() && prompt.bbox.is_none() {
            return Err(Error::Input("a prompt needs at least one point or a box".into()));
        }
        let dense = mask_logits.map(clamp_mask_logits).transpose()?;
        let traversed = self.traversed(object, frame);
        self.checkpoint(cancel, started)?;
        let d = self.decode_frame(object, frame, Some(prompt), dense.as_ref(), traversed.unwrap_or(false), traversed.is_some())?;
        self.checkpoint(cancel, started)?;
        let Decoded { for_memory, present, pointer, result, .. } = d;
        let pending = Pending { conditioning: traversed.is_none(), pointer: compact(&pointer)?, for_memory, present, result: Arc::new(result) };
        self.admit(pending_bytes(&pending))?;
        let mut next = self.objects.clone();
        next.entry(object).or_default().pending.insert(frame, pending);
        self.held.clear();
        Ok(Stage { info: TransitionInfo { kind: TransitionKind::Prompt, frame: Some(frame), object: Some(object), direction: traversed.map(|r| if r { Direction::Backward } else { Direction::Forward }) }, next, result: Some((object, frame, Slot::Pending)), clear_cache: false })
    }

    /// Meta's `add_new_mask`: the mask is the output; the decoder (one single-mask decode on the raw
    /// low features) only makes the pointer. The forced low-resolution scores pass the configured
    /// score processor before the memory input is made from them; the manual pixels stay the
    /// application's authority and no decoder scores are invented for them.
    fn build_dense(&mut self, frame: usize, object: ObjectId, mask: &Tensor, cancel: &Cancel, started: std::time::Instant) -> Result<Stage> {
        self.check_frame(frame)?;
        if mask.dims() != [1, 1, INPUT_SIZE, INPUT_SIZE] {
            return Err(Error::Input(format!("a mask prompt must be [1, 1, {INPUT_SIZE}, {INPUT_SIZE}], got {:?}", mask.dims())));
        }
        let values = mask.flatten_all()?.to_vec1::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Input("a mask prompt must be finite".into()));
        }
        let initial = self.traversed(object, frame).is_none();
        self.checkpoint(cancel, started)?;
        let feats = self.features(frame)?;
        let model = self.model.clone();
        let present = values.iter().any(|v| *v > 0.0);
        let high = ((mask * 20.0)? - 10.0)?;
        let forced = antialias_bilinear(&high.flatten_all()?.to_vec1::<f32>()?, (INPUT_SIZE, INPUT_SIZE), (4 * GRID, 4 * GRID))?;
        let down = mask.conv2d(&model.mask_down.0, 0, 4, 1, 1)?.broadcast_add(&model.mask_down.1.reshape((1, 1, 1, 1))?)?;
        let dense = model.image.prompt.dense_mask(&down)?;
        let out = model.image.decode_embedded(&feats.low, &feats.high, &model.image.prompt.sparse_none()?, &dense, false)?;
        self.checkpoint(cancel, started)?;
        let token = out.sam_tokens.narrow(1, 0, 1)?.squeeze(1)?;
        let decoder_present = scalar(&out.object_score)? > 0.0;
        let pointer = if decoder_present && present { model.pointer_proj.forward(&token)? } else { model.no_object_pointer.clone() };
        let low = self.process_mask_scores(&forced)?;
        let low_t = Tensor::from_vec(low.clone(), (1, 1, 4 * GRID, 4 * GRID), high.device())?;
        let for_memory = (low_t.upsample_bilinear2d(INPUT_SIZE, INPUT_SIZE, false)?.gt(0f32)?.to_dtype(DType::F32)? * 20.0)?;
        let for_memory = (for_memory - 10.0)?;
        let result = FrameMask { frame, object, logits: low, decoder_logits: None, iou: 1.0, object_score: if present { 10.0 } else { -10.0 }, occluded: !present };
        let pending = Pending { conditioning: initial, pointer: compact(&pointer)?, for_memory, present, result: Arc::new(result) };
        self.admit(pending_bytes(&pending))?;
        let mut next = self.objects.clone();
        next.entry(object).or_default().pending.insert(frame, pending);
        self.held.clear();
        Ok(Stage { info: TransitionInfo { kind: TransitionKind::Dense, frame: Some(frame), object: Some(object), direction: None }, next, result: Some((object, frame, Slot::Pending)), clear_cache: false })
    }

    /// Meta's `propagate_in_video_preflight`: encode every accepted pending output into memory and
    /// stage its merge into the consolidated banks (a conditioning frame's correction is dropped from
    /// the non-conditioning bank, as there). The staged memories and the pending inputs they come
    /// from are charged together.
    fn build_preflight(&mut self, cancel: &Cancel, started: std::time::Instant) -> Result<Stage> {
        let work: Vec<(ObjectId, usize)> = self.objects.iter().flat_map(|(id, o)| o.pending.keys().map(|f| (*id, *f))).collect();
        let mut staged: Vec<(ObjectId, usize, Arc<Memory>)> = Vec::with_capacity(work.len());
        for (id, frame) in &work {
            self.checkpoint(cancel, started)?;
            let feats = self.features(*frame)?;
            let p = self.objects.get(id).and_then(|o| o.pending.get(frame)).ok_or_else(|| Error::Compute("pending output vanished".into()))?;
            let (for_memory, present) = (p.for_memory.clone(), p.present);
            let memory = self.model.memorize(&feats.low, &for_memory, present)?;
            drop(feats);
            let memory = Memory { features: compact(&memory.features)?, pos: compact(&memory.pos)? };
            self.admit(tensor_bytes(&memory.features).saturating_add(tensor_bytes(&memory.pos)))?;
            staged.push((*id, *frame, Arc::new(memory)));
            self.held.clear();
        }
        self.checkpoint(cancel, started)?;
        let mut next = self.objects.clone();
        for (id, frame, memory) in staged {
            let Some(o) = next.get_mut(&id) else { continue };
            let Some(p) = o.pending.remove(&frame) else { continue };
            if p.conditioning {
                o.non_cond.remove(&frame);
                o.cond.insert(frame, FrameOut { memory, pointer: p.pointer, result: Some(p.result) });
            } else if !o.cond.contains_key(&frame) {
                o.non_cond.insert(frame, FrameOut { memory, pointer: p.pointer, result: Some(p.result) });
            }
        }
        for o in next.values_mut() {
            let cond: Vec<usize> = o.cond.keys().copied().collect();
            for f in cond {
                o.non_cond.remove(&f);
            }
        }
        Ok(Stage { info: TransitionInfo { kind: TransitionKind::Preflight, frame: None, object: None, direction: None }, next, result: None, clear_cache: false })
    }

    /// Meta's per-frame step of `propagate_in_video`: a conditioning frame returns its consolidated
    /// output (the traversal and its direction are still recorded, with no decode), any other frame
    /// is computed from the bank. One decode at most.
    fn build_advance(&mut self, frame: usize, object: ObjectId, direction: Direction, cancel: &Cancel, started: std::time::Instant) -> Result<Stage> {
        self.check_frame(frame)?;
        let reverse = direction == Direction::Backward;
        let info = TransitionInfo { kind: TransitionKind::Advance, frame: Some(frame), object: Some(object), direction: Some(direction) };
        let o = self.objects.get(&object).filter(|o| !o.cond.is_empty()).ok_or_else(|| Error::Input("no object has been prompted".into()))?;
        if o.cond.get(&frame).is_some_and(|f| f.result.is_some()) {
            let mut next = self.objects.clone();
            next.entry(object).or_default().tracked.insert(frame, reverse);
            return Ok(Stage { info, next, result: Some((object, frame, Slot::Cond)), clear_cache: false });
        }
        self.checkpoint(cancel, started)?;
        let d = self.decode_frame(object, frame, None, None, reverse, true)?;
        self.checkpoint(cancel, started)?;
        let memory = self.model.memorize(&d.feats.low, &d.for_memory, d.present)?;
        self.checkpoint(cancel, started)?;
        let Decoded { pointer, result, .. } = d;
        let memory = Memory { features: compact(&memory.features)?, pos: compact(&memory.pos)? };
        let entry = FrameOut { memory: Arc::new(memory), pointer: compact(&pointer)?, result: Some(Arc::new(result)) };
        self.admit(out_bytes(&entry))?;
        let mut next = self.objects.clone();
        let state = next.entry(object).or_default();
        state.non_cond.insert(frame, entry);
        state.tracked.insert(frame, reverse);
        self.held.clear();
        Ok(Stage { info, next, result: Some((object, frame, Slot::NonCond)), clear_cache: false })
    }

    /// Prepare, validate and commit in one go: success is acceptance. Returns the result mask.
    fn run_transition(&mut self, request: PrepareRequest<'_>, cancel: &Cancel) -> Result<Option<FrameMask>> {
        let prepared = self.prepare(request, cancel)?;
        let result = prepared.result().cloned();
        prepared.validate(cancel)?.commit();
        Ok(result)
    }

    /// Meta's preflight as one accepted transition (the official policy only).
    pub fn commit_pending(&mut self, cancel: &Cancel) -> Result<()> {
        if self.correction != CorrectionPolicy::Official {
            return Ok(());
        }
        self.run_transition(PrepareRequest::Preflight, cancel).map(|_| ())
    }

    fn advance_official(&mut self, id: ObjectId, frame: usize, direction: Direction, cancel: &Cancel) -> Result<FrameMask> {
        self.run_transition(PrepareRequest::Advance { frame, object: id, direction }, cancel)?.ok_or_else(|| Error::Compute("advance produced no mask".into()))
    }

    fn propagate_official(&mut self, only: Option<&[ObjectId]>, start: usize, reverse: bool, span: usize, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        self.op_started = Some(std::time::Instant::now());
        let r = self.propagate_official_inner(only, start, reverse, span, cancel, out);
        self.op_started = None;
        r
    }

    fn propagate_official_inner(&mut self, only: Option<&[ObjectId]>, start: usize, reverse: bool, span: usize, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        let n = self.frames.num_frames();
        let started = self.op_started.unwrap_or_else(std::time::Instant::now);
        self.commit_pending(cancel)?;
        let ids: Vec<ObjectId> = self.objects.iter().filter(|(id, o)| !o.cond.is_empty() && only.is_none_or(|l| l.contains(id))).map(|(id, _)| *id).collect();
        if ids.is_empty() {
            return Err(Error::Input("no object has been prompted".into()));
        }
        let direction = if reverse { Direction::Backward } else { Direction::Forward };
        for k in 0..=span {
            let frame = if reverse { start.checked_sub(k) } else { Some(start + k).filter(|f| *f < n) };
            let Some(frame) = frame else { break };
            for id in &ids {
                self.checkpoint(cancel, started)?;
                let mask = self.advance_official(*id, frame, direction, cancel)?;
                out(mask);
            }
        }
        Ok(())
    }

    fn propagate_each_official(&mut self, starts: &[(ObjectId, usize)], direction: Direction, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        self.op_started = Some(std::time::Instant::now());
        let r = self.propagate_each_inner(starts, direction, cancel, out);
        self.op_started = None;
        r
    }

    fn propagate_each_inner(&mut self, starts: &[(ObjectId, usize)], direction: Direction, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        let n = self.frames.num_frames();
        let started = self.op_started.unwrap_or_else(std::time::Instant::now);
        if starts.is_empty() {
            return Err(Error::Input("no object has been prompted".into()));
        }
        for (_, start) in starts {
            self.check_frame(*start)?;
        }
        // One accepted preflight, then the ordered traversal.
        self.commit_pending(cancel)?;
        for (id, _) in starts {
            if self.objects.get(id).is_none_or(|o| o.cond.is_empty()) {
                return Err(Error::Input("no object has been prompted".into()));
            }
        }
        for (id, frame) in pass_order(starts, n, direction == Direction::Backward) {
            self.checkpoint(cancel, started)?;
            let mask = self.advance_official(id, frame, direction, cancel)?;
            out(mask);
        }
        Ok(())
    }

    /// Drop memories that no later frame in either direction can still look at.
    fn prune(state: &mut ObjectState, frame: usize) {
        let keep = |f: usize, cond: &BTreeMap<usize, FrameOut>| f.abs_diff(frame) <= MAX_POINTERS || cond.keys().any(|c| f.abs_diff(*c) <= MEMORY_FRAMES);
        let doomed: Vec<usize> = state.non_cond.keys().copied().filter(|f| !keep(*f, &state.cond)).collect();
        for f in doomed {
            state.non_cond.remove(&f);
        }
    }

    fn check_frame(&self, frame: usize) -> Result<()> {
        if frame >= self.frames.num_frames() {
            return Err(Error::Input(format!("frame {frame} is outside the clip of {} frames", self.frames.num_frames())));
        }
        Ok(())
    }

    /// Prompt an object on a frame (clicks and/or a box). The frame becomes one of the object's
    /// conditioning frames when it has not been tracked yet; the mask comes back at once.
    pub fn prompt(&mut self, frame: usize, object: ObjectId, prompt: &Prompt) -> Result<FrameMask> {
        self.prompt_with_mask_logits(frame, object, prompt, None)
    }

    /// [`Self::prompt`] with the object's previous raw low-resolution mask logits (`[1, 1, 256, 256]`
    /// F32, finite) as the dense mask prompt, like the reference `add_new_points_or_box` using the
    /// preceding logits clamped to [-32, 32]. `None` is identical to [`Self::prompt`]. Only the
    /// prompted step uses it; propagation is unchanged.
    pub fn prompt_with_mask_logits(&mut self, frame: usize, object: ObjectId, prompt: &Prompt, mask_logits: Option<&Tensor>) -> Result<FrameMask> {
        if self.correction == CorrectionPolicy::Official {
            // Success is acceptance; hosts that must adopt the result first use `prepare`.
            return self.run_transition(PrepareRequest::Prompt { frame, object, prompt, mask_logits }, &Cancel::new())?.ok_or_else(|| Error::Compute("prompt produced no mask".into()));
        }
        self.check_frame(frame)?;
        prompt.validate()?;
        if prompt.points.is_empty() && prompt.bbox.is_none() {
            return Err(Error::Input("a prompt needs at least one point or a box".into()));
        }
        let dense = mask_logits.map(clamp_mask_logits).transpose()?;
        let tracked = self.objects.get(&object).is_some_and(|o| o.non_cond.contains_key(&frame));
        if tracked {
            // A correction on a tracked frame is conditioned on the memory like any tracked frame.
            let reverse = false;
            return self.step_with_mask(object, frame, Some(prompt), dense.as_ref(), reverse, false);
        }
        self.objects.entry(object).or_default();
        self.step_with_mask(object, frame, Some(prompt), dense.as_ref(), false, true)
    }

    /// Prompt an object on a frame with a mask: `[1, 1, 1024, 1024]` of 0 and 1 (see
    /// [`crate::resize::mask_to_model`]). The mask becomes the frame's output as it is (no decode of
    /// the mask itself); the decoder only makes the object pointer. Like a click prompt, the frame is a
    /// conditioning frame unless it has been tracked already.
    pub fn prompt_mask(&mut self, frame: usize, object: ObjectId, mask: &Tensor) -> Result<FrameMask> {
        if self.correction == CorrectionPolicy::Official {
            return self.run_transition(PrepareRequest::Dense { frame, object, mask }, &Cancel::new())?.ok_or_else(|| Error::Compute("mask prompt produced no mask".into()));
        }
        self.check_frame(frame)?;
        if mask.dims() != [1, 1, INPUT_SIZE, INPUT_SIZE] {
            return Err(Error::Input(format!("a mask prompt must be [1, 1, {INPUT_SIZE}, {INPUT_SIZE}], got {:?}", mask.dims())));
        }
        let values = mask.flatten_all()?.to_vec1::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Input("a mask prompt must be finite".into()));
        }
        let tracked = self.objects.get(&object).is_some_and(|o| o.non_cond.contains_key(&frame));
        self.objects.entry(object).or_default();
        self.step_mask(object, frame, mask, &values, !tracked)
    }

    /// The reference's `_use_mask_as_output`: the mask is the output, the decoder makes the pointer.
    fn step_mask(&mut self, id: ObjectId, frame: usize, mask: &Tensor, values: &[f32], conditioning: bool) -> Result<FrameMask> {
        let feats = self.features(frame)?;
        let model = self.model.clone();
        let present = values.iter().any(|v| *v > 0.0);
        let high = ((mask * 20.0)? - 10.0)?;
        let low = antialias_bilinear(&high.flatten_all()?.to_vec1::<f32>()?, (INPUT_SIZE, INPUT_SIZE), (4 * GRID, 4 * GRID))?;
        // The pointer: the decoder with the mask as its dense prompt and no clicks, on the frame's features
        // without the no-memory embedding.
        let down = mask.conv2d(&model.mask_down.0, 0, 4, 1, 1)?.broadcast_add(&model.mask_down.1.reshape((1, 1, 1, 1))?)?;
        let dense = model.image.prompt.dense_mask(&down)?;
        let out = model.image.decode_embedded(&feats.low, &feats.high, &model.image.prompt.sparse_none()?, &dense, true)?;
        let ious = out.iou.flatten_all()?.to_vec1::<f32>()?;
        let best = ious.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
        let token = out.sam_tokens.narrow(1, best, 1)?.squeeze(1)?;
        let decoder_present = scalar(&out.object_score)? > 0.0;
        let pointer = if decoder_present { model.pointer_proj.forward(&token)? } else { model.no_object_pointer.clone() };
        let pointer = if present { pointer } else { model.no_object_pointer.clone() };
        let memory = model.memorize(&feats.low, &high, present)?;
        let score = if present { 10.0 } else { -10.0 };
        let result = FrameMask { frame, object: id, logits: low, decoder_logits: None, iou: 1.0, object_score: score, occluded: !present };
        let state = self.objects.entry(id).or_default();
        let entry = FrameOut { memory: Arc::new(memory), pointer, result: conditioning.then(|| Arc::new(result.clone())) };
        if conditioning {
            state.non_cond.remove(&frame);
            state.cond.insert(frame, entry);
        } else {
            state.non_cond.insert(frame, entry);
            Self::prune(state, frame);
        }
        self.check_budget()?;
        Ok(result)
    }

    /// Track every object from `start` through `max_frames` frames (the rest of the clip when
    /// `None`) in `direction`, calling `out` for each object on each frame as it is finished.
    pub fn propagate(&mut self, start: usize, direction: Direction, max_frames: Option<usize>, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        self.propagate_objects(None, start, direction, max_frames, cancel, out)
    }

    /// [`VideoTracker::propagate`] for the listed objects only (`None`: every prompted object).
    pub fn propagate_objects(
        &mut self,
        only: Option<&[ObjectId]>,
        start: usize,
        direction: Direction,
        max_frames: Option<usize>,
        cancel: &Cancel,
        out: &mut dyn FnMut(FrameMask),
    ) -> Result<()> {
        self.check_frame(start)?;
        let n = self.frames.num_frames();
        let reverse = direction == Direction::Backward;
        let span = max_frames.unwrap_or(n).min(n);
        if self.correction == CorrectionPolicy::Official {
            return self.propagate_official(only, start, reverse, span, cancel, out);
        }
        let ids: Vec<ObjectId> = self.objects.iter().filter(|(id, o)| !o.cond.is_empty() && only.is_none_or(|l| l.contains(id))).map(|(id, _)| *id).collect();
        if ids.is_empty() {
            return Err(Error::Input("no object has been prompted".into()));
        }
        let started = std::time::Instant::now();
        for k in 0..=span {
            let frame = if reverse { start.checked_sub(k) } else { Some(start + k).filter(|f| *f < n) };
            let Some(frame) = frame else { break };
            for id in &ids {
                if cancel.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                if let Some(limit) = self.budget.max_seconds.filter(|l| started.elapsed().as_secs_f64() > *l) {
                    return Err(Error::Budget(format!("tracking took more than {limit} s")));
                }
                let mask = self.mask_at(*id, frame, reverse)?;
                out(mask);
            }
        }
        Ok(())
    }

    /// Object `id` on `frame`: the result stored with its prompt there, else the tracker's step.
    fn mask_at(&mut self, id: ObjectId, frame: usize, reverse: bool) -> Result<FrameMask> {
        let stored = self.objects.get(&id).and_then(|o| o.cond.get(&frame)).and_then(|o| o.result.as_deref().cloned());
        match stored {
            Some(m) => Ok(m),
            None => self.step(id, frame, None, reverse, false),
        }
    }

    /// Track every listed object from its own prompted frame (`starts`: object, frame) in one pass
    /// in `direction`, frame by frame with all the objects that have started: a frame is finished
    /// for all of them before the next begins, so a caller that assembles the objects of a frame
    /// does not hold frames waiting for a pass that has not got there. A backward pass leaves out
    /// each object's own start frame (the forward pass reported it).
    pub fn propagate_each(&mut self, starts: &[(ObjectId, usize)], direction: Direction, cancel: &Cancel, out: &mut dyn FnMut(FrameMask)) -> Result<()> {
        if self.correction == CorrectionPolicy::Official {
            return self.propagate_each_official(starts, direction, cancel, out);
        }
        let n = self.frames.num_frames();
        for (id, start) in starts {
            self.check_frame(*start)?;
            if self.objects.get(id).is_none_or(|o| o.cond.is_empty()) {
                return Err(Error::Input("no object has been prompted".into()));
            }
        }
        if starts.is_empty() {
            return Err(Error::Input("no object has been prompted".into()));
        }
        let reverse = direction == Direction::Backward;
        let started = std::time::Instant::now();
        for (id, frame) in pass_order(starts, n, reverse) {
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if let Some(limit) = self.budget.max_seconds.filter(|l| started.elapsed().as_secs_f64() > *l) {
                return Err(Error::Budget(format!("tracking took more than {limit} s")));
            }
            let mask = self.mask_at(id, frame, reverse)?;
            out(mask);
        }
        Ok(())
    }
}

/// A previous-mask input validated and clamped to [-32, 32] where it is consumed.
fn clamp_mask_logits(m: &Tensor) -> Result<Tensor> {
    if m.dims() != [1, 1, 4 * GRID, 4 * GRID] || m.dtype() != DType::F32 {
        return Err(Error::Input(format!("mask logits must be F32 [1, 1, {0}, {0}], got {1:?} {2:?}", 4 * GRID, m.dtype(), m.dims())));
    }
    if m.flatten_all()?.to_vec1::<f32>()?.iter().any(|v| !v.is_finite()) {
        return Err(Error::Input("mask logits must be finite".into()));
    }
    Ok(m.clamp(-32f32, 32f32)?)
}

/// The (object, frame) steps of [`VideoTracker::propagate_each`], in order: frame by frame from the
/// first start (the last, backwards), each frame for the objects whose pass has reached it.
fn pass_order(starts: &[(ObjectId, usize)], frames: usize, reverse: bool) -> impl Iterator<Item = (ObjectId, usize)> + '_ {
    let first = if reverse { starts.iter().map(|s| s.1).max() } else { starts.iter().map(|s| s.1).min() };
    let span = match first {
        Some(f) if reverse => f + 1,
        Some(f) => frames.saturating_sub(f),
        None => 0,
    };
    (0..span).flat_map(move |k| {
        let frame = if reverse { first.unwrap_or(0).saturating_sub(k) } else { first.unwrap_or(0) + k };
        starts.iter().filter_map(move |(id, start)| {
            let reached = if reverse { *start > frame } else { *start <= frame };
            reached.then_some((*id, frame))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(starts: &[(u32, usize)], frames: usize, reverse: bool) -> Vec<(u32, usize)> {
        let starts: Vec<(ObjectId, usize)> = starts.iter().map(|(o, s)| (ObjectId(*o), *s)).collect();
        pass_order(&starts, frames, reverse).map(|(o, f)| (o.0, f)).collect()
    }

    /// Forward, every frame is finished for all the objects that have started before the next frame
    /// begins; an object joins the pass at its own start frame.
    #[test]
    fn the_forward_pass_goes_frame_by_frame_over_the_objects_that_have_started() {
        assert_eq!(order(&[(1, 0), (2, 3)], 6, false), [(1, 0), (1, 1), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 5), (2, 5)]);
        // Prompts on the same frame go together; the pass starts at the earliest start.
        assert_eq!(order(&[(1, 2), (2, 2)], 4, false), [(1, 2), (2, 2), (1, 3), (2, 3)]);
        assert_eq!(order(&[(1, 2)], 4, false), [(1, 2), (1, 3)]);
    }

    /// Backward, from the last start down to frame 0, leaving out each object's own start frame.
    #[test]
    fn the_backward_pass_goes_frame_by_frame_and_skips_each_objects_own_start() {
        assert_eq!(order(&[(1, 0), (2, 3)], 6, true), [(2, 2), (2, 1), (2, 0)]);
        assert_eq!(order(&[(1, 2), (2, 3)], 6, true), [(2, 2), (1, 1), (2, 1), (1, 0), (2, 0)]);
        assert!(order(&[(1, 0)], 6, true).is_empty(), "nothing before frame 0");
    }

    /// A frame is finished for every object before the next begins, however far apart the prompts
    /// are: the number of frames waiting for an object never grows with the clip.
    #[test]
    fn no_frame_waits_for_an_object_that_has_not_reached_it() {
        let steps = order(&[(1, 0), (2, 1000)], 2000, false);
        let mut waiting = 0usize;
        let mut worst = 0usize;
        let mut seen = std::collections::BTreeMap::<usize, usize>::new();
        for (_, f) in steps {
            let n = seen.entry(f).or_default();
            *n += 1;
            // From frame 1000 on, both objects report each frame back to back.
            if f >= 1000 && *n == 2 {
                waiting -= 1;
            } else if f >= 1000 {
                waiting += 1;
            }
            worst = worst.max(waiting);
        }
        assert!(worst <= 1, "{worst} frames waiting");
    }
}
