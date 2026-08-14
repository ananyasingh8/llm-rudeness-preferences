# Local LLM Runtime Research

Date: 2026-08-14

Status: Research complete. Runtime compatibility tests are not complete.

## Purpose

This document compares local model runtimes for this project.

The first target model is Gemma 4 E2B.

The local computer has one NVIDIA RTX 4090 GPU. The GPU has 24 GB of VRAM.

The runtime must meet two main needs:

1. It must be easy to run on the local computer.
2. It must let us read or change model activations during inference.

High throughput is useful. It is not the first need.

## Software Layers

These tools are not all alternatives. Some tools are on different layers of one stack.

```text
Research code
  |
  +-- NNsight or TransformerLens       activation research tools
  |
Model implementation
  |
  +-- Transformers                     Gemma 4 Python model and processors
  |
Compute framework
  |
  +-- PyTorch                          tensors, CUDA operations, and gradients
  |
GPU software
  |
  +-- CUDA                             NVIDIA GPU execution
```

An optimized serving stack can replace some middle layers:

```text
Research code or application
  |
  +-- NNsight vLLM backend             optional activation research layer
  |
  +-- vLLM                             model execution, batching, and serving
  |
  +-- PyTorch, Triton, and CUDA        lower execution dependencies
```

Ollama and LM Studio are local applications. LiteLLM is a gateway. They are above an inference engine.

### PyTorch

PyTorch is a compute framework. It supplies tensors, GPU operations, neural-network modules, and automatic differentiation.

PyTorch does not contain Gemma 4 model code by itself. A developer can write that code with PyTorch, or use a model library that already contains it.

### Transformers

Transformers is a model library. Hugging Face develops it.

Transformers supplies the current Gemma 4 Python model, configuration classes, tokenizer integration, processors, generation code, and model-output types. Its Gemma 4 implementation uses PyTorch for calculation.

Therefore, this document does not compare PyTorch against Transformers. The first candidate is one stack: Transformers on PyTorch.

A useful comparison is `Transformers + PyTorch` against `vLLM`, `SGLang`, or `llama.cpp`. These options execute the model through different graphs and kernels.

### NNsight

NNsight is a research instrumentation library. It is not a model and it is not primarily an inference engine.

NNsight wraps a model that another system executes. Its trace API can select a module input or output. Research code can save that value or replace it during the forward pass.

With a PyTorch model, NNsight can also use gradients. With its vLLM backend, NNsight can process more prompts but does not support gradients.

NNsight keeps the underlying model structure. This is useful for a new model such as Gemma 4 because research code can access the actual model modules. The exact module paths still need a compatibility test.

### TransformerLens

TransformerLens is a mechanistic-interpretability library. It supplies standard hook points, activation caches, activation patching tools, and common names for transformer components.

TransformerLens is more opinionated than NNsight. It maps supported model architectures into a common research interface. This can make experiments easier to write and compare.

The mapping is also a compatibility boundary. A new architecture needs a correct adapter or bridge. We must compare its Gemma 4 output with the documented Transformers model before we use it for a paper.

### NNsight Compared With TransformerLens

| Question | NNsight | TransformerLens |
|---|---|---|
| Main purpose | Inspect or change an existing model execution | Give standard tools and hook names for interpretability |
| Underlying model | Usually keeps the original PyTorch model | Uses a supported model mapping or bridge |
| New architecture support | Often possible before a special adapter exists | Usually needs a correct architecture mapping |
| Hook names | Follow the actual model module structure | Use standardized interpretability names |
| Gradients | Yes with the PyTorch backend | Yes for supported PyTorch paths |
| High-throughput option | Has a vLLM backend | Not its main purpose |
| Main risk | Model-specific paths and trace behavior | Mapping coverage and numerical parity |

For this project, NNsight is the first research-tool candidate. TransformerLens remains useful if its Gemma 4 E2B mapping passes the parity tests.

## Short Answer

No one runtime is best for all tasks.

Use Transformers on PyTorch for the reference path, then add NNsight for activation work. Evaluate NNsight with vLLM for larger prompt sets and llama.cpp as a separate Q4 GGUF deployment reference. See [Current Recommendation](#current-recommendation) for the ordered decision table.

## Hugging Face Is Not Required

Hugging Face can supply model files. It can also supply Python model code, tokenizers, processors, and chat templates.

These are separate functions. The project does not need to use all of them.

We can get the same approved weights from Hugging Face, Kaggle, Google storage, or another verified source. The file revision and checksum are more important than the download service.

NNsight needs a PyTorch model object. It does not require the Hugging Face website. The current Transformers package has a documented Gemma 4 PyTorch implementation.

llama.cpp needs a compatible GGUF file. It does not need Transformers to run that file.

vLLM needs a supported model implementation and weight format. It can download files from Hugging Face, or it can use local files.

### Local Transformers Files

A local Transformers model directory can remove the network dependency. The required file set depends on the model and the selected input types.

The loader needs the applicable configuration and checkpoint files. Text use also needs the applicable tokenizer files. Multimodal use can need processor files. A sharded Safetensors checkpoint usually has an index file.

For a repository-backed artifact, pin the resolved commit SHA. For a local path, record identity separately. The project must compute and record a SHA-256 checksum for each local file. Load the directory with offline-only loading enabled during a reproducibility run.

### Local GGUF Files

A text-only llama.cpp run needs the selected `.gguf` model file. Extra multimodal artifacts are format, model, and runtime dependent. The inspected Gemma 4 source does not prove a separate projector requirement.

Pin the llama.cpp commit. The project must verify the model-file checksum before loading. llama.cpp does not enforce the project checksum. Pass the local file path to the command-line tool or C API.

## Suggested Runtime Roles For This Project

There is no one industry-standard runtime for all scientific work.

Use PyTorch and Transformers for model-internal research. Add NNsight or TransformerLens for activation experiments.

Use vLLM for high-throughput text generation and benchmark runs. Add the NNsight vLLM backend when many prompts need the same activation operation.

Use SGLang when prefix reuse, agent control, or structured generation is the main workload.

Use llama.cpp for portable GGUF and low-memory local inference.

Use Ollama and LM Studio for easy local applications and demonstrations. Do not use their public APIs for activation research.

Use LiteLLM only as an API gateway.

## Important Result For Gemma 4 E2B

The [Gemma 4 technical report](https://arxiv.org/abs/2607.02770) reports approximately 5.1 billion stored parameters and 2.3 billion effective parameters for each token. It attributes much of the difference to Per-Layer Embeddings.

The estimated BF16 weight storage is 10.2 decimal GB: 5.1 billion parameters multiplied by 2 bytes. This is a calculation, not a measured result. A 24 GB GPU has additional capacity, but the context and activation limits still require measurement.

The 10.2 GB value is raw parameter storage only. It does not include the CUDA context, temporary buffers, activations, multimodal encoders, or KV cache.

We do not yet have measured Gemma 4 E2B memory values for this computer. Measure model load, one-token generation, selected context lengths, KV-cache use, and each saved activation set. Measure text, image, and audio paths separately.

Q4 weights use much less memory. However, Q4 does not mean that all activations use 4 bits. Many Q4 runtimes use 16-bit activations.

The raw weight estimate makes BF16 a candidate for the 24 GB GPU. Do not select it for the main study until the exact checkpoint loads and measured VRAM leaves enough capacity for the selected context, KV cache, and saved activations. Keep Q4 as a fallback and deployment comparison.

## What GGUF Is

GGUF is a binary model-file format from the ggml ecosystem. llama.cpp is a major user of this format.

### What The ggml Ecosystem Is

ggml is a C tensor library and an execution framework. It is designed for local machine-learning inference. It has few runtime dependencies.

The ggml ecosystem is the group of related libraries, formats, tools, and applications that use ggml concepts or code.

Important parts include:

- `ggml`: Tensor operations, computation graphs, memory control, and hardware backends.
- `llama.cpp`: An LLM inference engine built with ggml technology.
- `GGUF`: The current model-file format used by llama.cpp and related tools.
- Quantization types: Formats such as Q4_0 and Q4_K_M for low-bit tensors.
- Hardware backends: CPU, CUDA, Metal, Vulkan, and other execution paths.
- Language bindings: Python and other interfaces to native libraries.
- Applications: Ollama, LM Studio, and other tools that use llama.cpp or compatible code.

The ecosystem is not one package. A tool can support GGUF without exposing the ggml graph. A tool can also use llama.cpp internally and hide most llama.cpp settings.

The name `GGML` also referred to an older model-file format. GGUF replaced that older format. In this document, `ggml` means the tensor library and project family. `GGUF` means the current file format.

ggml and PyTorch solve some similar low-level problems. Both can run tensor operations on hardware. PyTorch focuses on broad model development, training, automatic differentiation, and Python workflows. ggml focuses on small native inference deployments, explicit memory control, and portable quantized execution.

The name does not describe one quantization method. A GGUF file can contain BF16, FP16, Q8, Q6, Q5, Q4, or other tensor types.

A GGUF file can contain these items:

- Model tensors, such as weights and embeddings.
- Tensor type and tensor shape data.
- Model architecture data.
- Tokenizer data.
- Special token data.
- A chat template.
- Context and rope settings.
- Other model metadata.

This design makes one file easy to copy. A compatible runtime can read the file without a separate Python model package. The runtime must support the model architecture and each required input type.

GGUF is not an execution engine. llama.cpp, Ollama, and LM Studio are examples of software that can load GGUF files.

GGUF is not equal to Q4. Q4 is one possible tensor precision in a GGUF file.

### GGUF File Layout

A GGUF file has a defined binary layout.

The header identifies the file as GGUF. It also gives the GGUF version, tensor count, and metadata count.

The metadata section stores typed key and value pairs. A value can be a number, text, or an array. Model architecture, tokenizer, and chat-template data are examples of metadata.

The tensor-information section gives each tensor name, shape, type, and data offset.

The tensor-data section stores the tensor bytes. `general.alignment` controls alignment. The GGUF specification uses 32 when that metadata key is absent. Tensor offsets are relative to the tensor-data section.

A quantized tensor stores blocks of low-bit values. It also stores scale data or other block data. The runtime uses this data to calculate an approximate high-precision value.

GGUF does not define the complete inference algorithm. The runtime still supplies graph code, CPU or GPU kernels, memory control, and token generation.

### GGUF Strengths

- One file can contain the weights and most required metadata. Some multimodal deployments can still need external artifacts.
- llama.cpp has strong GGUF support.
- GGUF supports mixed tensor types.
- GGUF works on CPUs and many GPU types.
- A Q4 GGUF file uses less memory than a BF16 model.
- A repository revision, file manifest, and external checksums can identify one exact artifact. Embedded GGUF metadata does not prove byte identity or provenance.

### GGUF Weaknesses

- PyTorch cannot use a GGUF file as a normal PyTorch model.
- PyTorch forward hooks cannot attach to a llama.cpp graph.
- NNsight cannot use the GGUF graph as a normal PyTorch graph.
- Gradients are not normally available.
- Activation access needs llama.cpp C or C++ callbacks or custom code.
- Tensor names can differ from names in the original training model.
- Fused operations can remove some intermediate values.
- GGUF conversion and quantization can change model output.

### GGUF And Activations

llama.cpp has an `eval-callback` example. The callback can show graph operations and tensor data during inference.

This debug example proves that tensor observation is possible. It does not demonstrate safe tensor replacement. It is not a stable high-level research API. It needs C++ code and knowledge of the llama.cpp graph. Data transfer from the GPU to the CPU can also make inference much slower.

Use GGUF when memory use and easy deployment are the main needs. Do not make GGUF the only model format when activation research is a main need.

## What Google QAT Is

QAT means Quantization-Aware Training. The definitions and Gemma 4 artifact claims in this section come from the cited Google technical report and model cards. The cloned runtime code does not prove the model's training history.

Normal training uses high-precision values. A later conversion can change these values to low-precision values. This later conversion is Post-Training Quantization, or PTQ.

PTQ can reduce model quality. The model does not learn about the later rounding errors during training.

QAT simulates low-precision effects during training. The training process can adjust the weights for these effects. The final low-precision model can keep more of the original model quality.

Google supplies Gemma 4 QAT checkpoints for Q4_0. Google states that these checkpoints keep quality close to BF16 and use less memory. We have not reproduced the cited quality or memory results.

Google supplies four main forms. QAT is the training method. Each row below is a model artifact from that method.

| Form | Selected artifact | Claimed weight precision | Activation precision | Main runtime |
|---|---|---:|---:|---|
| Unquantized QAT checkpoint | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | Half precision | Set by runtime | Research and custom compilation |
| GGUF Q4_0 | `google/gemma-4-E2B-it-qat-q4_0-gguf` | Q4_0 | Set by runtime | llama.cpp and related tools |
| Mobile `wNa8o8` | Select from the official Google QAT collection | Mixed low-bit | Claimed 8-bit | Mobile runtime selected by Google artifact |
| Compressed Tensors `w4a16` | `google/gemma-4-E2B-it-qat-w4a16-ct` | Claimed 4-bit | Verify from artifact config | vLLM or supported Transformers path |

The artifact names and precision claims require inspection of the downloaded artifact. For Compressed Tensors, inspect `config.json`, `config_groups`, weight schemes, activation schemes, ignored modules, and `kv_cache_scheme`.

The KV-cache precision is a separate setting. A model-specific quantization configuration can prescribe it, or a runtime can select it separately.

The unquantized QAT checkpoint is intended to provide a PyTorch-compatible research path. Exact loading and output behavior remain untested. It does not give the Q4 memory reduction until a later compiler or quantizer packs the weights.

The GGUF Q4_0 file is important for deployment tests. It should reduce weight storage relative to BF16. Exact load memory and runtime memory remain unmeasured. It does not give a normal PyTorch module graph.

The current implementation uses Google's official instruction-tuned GGUF Q4_0 artifact. Base or non-instruction-tuned support is deferred because Google does not currently publish an official base Q4_0 artifact.

The `ggml-org/gemma-4-E2B-GGUF` repository provides a separate base/non-instruction-tuned llama.cpp path. It is an automatic ggml-org conversion of `google/gemma-4-E2B` revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`. Repository revision `0d85c394df5e9f3c5b07c791e31680e399042acc` contains BF16 and Q8_0 GGUF files, but no Q4_0 file. The Q8_0 file is a post-training conversion, not a Google QAT artifact.

Pinned base files at that revision:

| File | Size | LFS SHA-256 |
|---|---:|---|
| `gemma-4-E2B-BF16.gguf` | 9,311,286,336 bytes | `11111065543435393ecfe54d3b6f8ab106617c306666e2fa882ef48dd87d5331` |
| `gemma-4-E2B-Q8_0.gguf` | 4,967,478,336 bytes | `fcf7224c47518fca8a8a85391cfb0873a94d3dee4434fb77763113bfa7264d97` |

The Compressed Tensors `w4a16` file is a candidate for vLLM tests. The cited sources describe Compressed Tensors schemas, but they do not prove W4A16 kernel support or compatibility with the selected Gemma artifact. Inspect the artifact configuration and test loading with the selected vLLM revision.

## End-To-End Pipeline

The pipeline has two data flows. The first flow prepares the model. The second flow runs an experiment. The execution diagram is conceptual. A specific engine can combine stages or schedule them in a different internal order.

```text
MODEL PREPARATION

[Artifact source]
       |
       v
[Weights + config + tokenizer + processor]
       |
       v
[Revision pin + checksums + local cache]
       |
       v
[Loader and model implementation]


EXPERIMENT EXECUTION

[Experiment code or user]
            |
            v
    [Gateway or client]
            |
            v
       [Model server]
            |
            v
    [Prompt or media]
            |
            v
[Chat template + tokenizer + processor]
            |
            v
 [Token IDs and media tensors]
            |
            v
  [Scheduler + KV cache] -------------------------+
                                                  |
 [KV cache] ---------------- read ----------------+--> [Model graph] <------ [Activation instrumentation]
                                                          |                         |
                                                          v                         +-- read
                                                  [CPU or GPU kernels]               +-- change
                                                     |          |
                                                     |          +-- write --> [KV cache]
                                                     v
                                                 [Logits] -> [Sampler] -> [Output tokens]
                                       |
                                       v
                              [Model server response]


EVIDENCE FLOW

[Hashes + versions + inputs + activations + outputs + metrics]
       |
       v
[Experiment record]
```

Each layer has a different decision. Do not select a gateway when the decision concerns tensor execution. Do not select a file format when the decision concerns activation hooks.

The tables below compare alternatives only inside one layer.

Legend for capability cells:

- `Yes`: The option has a normal supported path.
- `Partial`: The path has limits or needs extra work.
- Unless a cell says that a local test passed, capability statements describe the inspected source snapshot. They do not prove compatibility with the exact Gemma 4 E2B checkpoint.
- `No`: The option does not provide this function.
- `Unknown`: The exact Gemma 4 E2B path needs a test.

Local source citations in the layer sections are relative to `/home/minttea/codebases/llms/`.

## Layer 1: Artifact Source And Identity

This layer gets files and proves which files the experiment used.

```text
[Google, Hugging Face, Kaggle, or another source]
                         |
                         v
              [Versioned model files]
                         |
                         v
              [Local verified artifact]
                   |             |
                   +-- revision  +-- SHA-256
```

| Source method | Exact version support | Offline after download | Main strength | Main weakness |
|---|---:|---:|---|---|
| Local verified directory | Yes | Yes | Full control of the experiment input | The project must manage updates |
| Versioned model registry | Yes | Yes, if all files are cached and offline-only loading is enabled | Convenient download and metadata | The client and cache add dependencies |
| Moving model tag or branch | Partial | Yes | Easy initial use | Later downloads can change |
| Unversioned direct URL | No | Yes | Simple transfer | A URL is insufficient identity without immutable version data and an external checksum |

The selected source service is not part of model execution. Copying identical bytes from a different service does not change the model artifact.

Unsloth's model mapper recognizes `unsloth/...` Gemma 4 quantization identifiers and associates them with corresponding Unsloth and Google model identifiers. Treat an `unsloth/...` quantization as an Unsloth-prepared derivative, not as a Google-official artifact. Record both the derivative revision and its declared upstream identity. Namespace mapping does not prove publication, artifact contents, byte identity, or numerical parity.

Required output from this layer:

- Local file paths.
- Repository identifier and artifact file name.
- Resolved immutable commit SHA for each repository-backed file.
- Separate model, tokenizer, processor, and projector revisions when applicable.
- Complete file manifest with file sizes and SHA-256 checksums.
- Loader, conversion, and quantization tool revisions.
- Local-directory, cache-snapshot, or network-download source mode.
- Model-specific license, usage restrictions, access status, and attribution.

The project experiment harness owns these records. Runtime caches and embedded GGUF metadata do not prove artifact identity.

Local evidence: `transformers/src/transformers/modeling_utils.py:3867-3869,3891-3914,3936-3944`; `vllm/vllm/config/model.py:567-592`; `llama.cpp/tools/cli/cli-context.cpp:110-115`; `unsloth/unsloth/models/mapper.py:25-44`.

## Layer 2: Artifact Format And Quantization

This layer defines how the model values are stored.

```text
[Trained model]
       |
       +--> [BF16 or FP16 Safetensors]
       |
       +--> [QAT half-precision checkpoint]
       |
       +--> [Compressed Tensors w4a16]
       |
       +--> [GGUF Q4_0]
       |
       +--> [Mobile wNa8o8]
```

| Artifact form | Weight storage | Activation contract | Container structure | Best use |
|---|---|---|---|---|
| Standard Safetensors | BF16 or FP16 | Set by runtime | Tensor files; model configuration is separate | Reference behavior and gradients |
| Unquantized QAT checkpoint | Claimed half precision; verify artifact | Set by runtime | Tensor files plus separate config files | QAT research and later compilation |
| Compressed Tensors `w4a16` | Verify artifact `config.json` | Activation scheme is independently configured | Safetensors plus a quantization schema in configuration | Quantized GPU serving |
| GGUF Q4_0 | 4-bit blocks | Set by runtime | One main model file for text-only use; multimodal deployments can require external artifacts | Portable local deployment |
| Mobile `wNa8o8` | Externally claimed mixed low-bit | Externally claimed 8-bit | Verify the Google mobile package | Mobile deployment |

This layer does not define activation access. The execution and instrumentation layers define that access.

Unsloth can export merged 16-bit, merged 4-bit, LoRA, and GGUF artifacts. Export creates a new artifact for this layer. It does not make Unsloth the engine that later executes GGUF; llama.cpp remains the separate native execution path.

Required output from this layer:

- Exact artifact form.
- Weight precision for each module group.
- Activation and accumulation precision.
- KV-cache precision.
- Quantization calibration or QAT source.

For Compressed Tensors, also record `format`, `quantization_status`, `config_groups`, targets, group size, symmetry, zero points, scales, ignored modules, input/output activation schemes, and `kv_cache_scheme`.

For GGUF, also record the GGUF version, byte order, alignment, tensor types, architecture metadata, and external multimodal artifacts.

Local evidence: `ggml/docs/gguf.md:147-158,262-353,377-456`; `compressed-tensors/README.md:104-165`; `compressed-tensors/src/compressed_tensors/quantization/quant_scheme.py:26-63,104-131`; `transformers/src/transformers/utils/quantization_config.py:1096-1174`; `unsloth/unsloth_cli/commands/export.py:12-14,47-62,94-129`.

## Layer 3: Input Processing And Model Definition

This layer changes user input into model input. It also defines the model modules and forward pass. The diagram shows the verified `Gemma4UnifiedProcessor` pattern. Other Gemma 4 classes can use different input paths.

```text
[Messages] -> [Chat template] -> [Text tokenizer] -> [Token IDs] -----+
                                                                       |
[Image or audio] -> [Media processor] -> [Media tensors] --------------+--> [Combined model input]
                                                                              |
                                                                              v
                                                                     [Model definition]
```

| Definition form | Chat-template source | Tokenizer source | Media processing | Module representation | Main risk |
|---|---|---|---|---|---|
| Python model package | Separate config files | Separate tokenizer files | Python processor modules | Python module tree | Package and model revision must match |
| Self-describing native artifact | Artifact metadata | Artifact metadata | Model- and format-dependent external processing | Native graph | Metadata can omit a required input path |
| Compiled engine package | Build configuration | Pinned external tokenizer | Build-specific processor | Compiled graph | A rebuild is needed after graph changes |

The serialized prompt is an experiment input. Save the text after the chat template runs. Also save the final token IDs.

Transformers implements text, image, video, and audio processor branches. llama.cpp maps a 35-layer Gemma 4 configuration to E2B. SGLang includes an Intel-XPU text smoke test. vLLM defines Gemma 4 classes, and TensorRT-LLM maps Gemma 4 architecture names to implementations. None of these citations proves successful E2B multimodal execution for the selected artifact.

Local evidence: `transformers/src/transformers/models/gemma4_unified/processing_gemma4_unified.py:136-158,176-227,273-304`; `llama.cpp/src/models/gemma4.cpp:3-28,145-155`; `sglang/test/registered/xpu/test_gemma_4_e2b.py:1-56`; `vllm/vllm/model_executor/models/gemma4.py:214-225,949,1494-1506`; `TensorRT-LLM/tensorrt_llm/_torch/models/_arch_index.py:54-57`.

## Layer 4: Tensor Execution

This layer calculates the forward pass.

```text
[Model modules or native graph]
              |
              v
      [Tensor operations]
              |
              v
 [Execution graph and operations]
              |
              v
 [Compiled, JIT, or backend kernels]
              |
              v
            [Logits]
```

| Execution option | Evidence in cited snapshot | Gradient status | Open validation |
|---|---|---|---|
| PyTorch | General tensor framework with CPU and GPU execution | Automatic differentiation is part of the framework | Exact Gemma 4 path and compiled behavior |
| Triton | JIT compiler and language for custom GPU kernels | Not established by the cited range | Integration with each model runtime |
| ggml | Native tensor graph with multiple backends | Automatic differentiation is advertised by ggml | Gradient use in the selected LLM path |
| llama.cpp | Native CPU/GPU model inference through ggml backends | No normal exposed inference gradient path found | Exact Gemma 4 artifact execution |
| vLLM | Inference and serving system designed for high throughput | Normal inference gradient path not established | Exact execution mode and artifact compatibility |
| SGLang | Not established by the current Layer 4 citation set | Not established | Add execution-path evidence before comparison |
| TensorRT-LLM | Separate TensorRT and PyTorch-oriented execution components | Public inference gradient path not established | Exact E2B engine and output path |

Compilation and fusion are different. Compilation creates executable graph or kernel code. Fusion combines operations. Fusion can prevent an intermediate tensor from being materialized. Prefer an unfused eager path or an explicitly instrumented graph when an experiment must observe that value.

Local evidence: `pytorch/README.md:62-65,203-211`; `triton/README.md:25,219-240`; `ggml/README.md:12-17`; `llama.cpp/README.md:53-75`; `vllm/README.md:24,33-38`; `TensorRT-LLM/.github/tava_architecture_diagram.md:20-49`.

## Layer 5: Activation Instrumentation

This layer observes or changes values inside the model graph.

```text
[Module input] -> [Input hook point] -> [Module] -> [Output hook point] -> [Module output]
                       ^                                 ^
           read/change |                                 | read/change
                       +-------- [Experiment code] -------+
                                      |
                                      v
                              [Saved activation]

[Loss] -- backward pass --> [Gradient at selected hook, if supported]
```

| Instrumentation option | Execution target | Read | Change | Gradients | Standard hook names | Generation-step access | Gemma 4 E2B status |
|---|---|---:|---:|---:|---:|---:|---|
| Native PyTorch hooks | PyTorch model | Yes | Yes | Not established by the cited forward-hook range | No | Hook fires on each module invocation | Model path is documented; exact hooks need tests |
| NNsight | PyTorch model | Yes | Yes | Not established by the cited ranges | No | Dedicated generation tracing | Unknown until local checkpoint test |
| nnterp | NNsight model | Delegated to NNsight; not established by this citation | Delegated to NNsight; not established by this citation | Delegated to NNsight; not established by this citation | Yes | Backend- and workflow-dependent | Mapping test required |
| TransformerLens | Supported PyTorch path | Yes for tested residual hooks | Yes for tested residual hooks | Not established by the cited ranges | Yes | Explicit loop or supported wrapper required | Gemma 4 adapter and small-fixture integration tests exist; official E2B parity and requested hook coverage remain unknown |
| NNsight vLLM integration | vLLM | Yes | Yes | No | No | Instruments scheduled forward steps | Text-oriented documented path; exact checkpoint test required |
| llama.cpp eval-callback debug example | llama.cpp graph | Callback registration is demonstrated; readable values are not established by this range | Not established | No normal inference gradient path | No | Runs during graph evaluation | Intervention is not established by the cited range |
| TensorRT-LLM selected outputs | TensorRT graph | Selected logits and configured outputs | Not established by public API | No public inference gradient path | No | Selected generation outputs only | Arbitrary intermediate hooks require model, plugin, or build work |

This is the main comparison layer for the current research need. NNsight and TransformerLens belong here. PyTorch supplies the underlying hook and gradient system.

Local evidence: `pytorch/torch/nn/modules/module.py:1629-1712`; `nnsight/src/nnsight/intervention/envoy.py:178-205,323-358,483-490`; `nnsight/src/nnsight/modeling/vllm/README.md:28-36,186-221,302-313,377-401`; `nnsight/src/nnsight/modeling/vllm/intervention-gaps/REPORT.md:166-174,234`; `nnterp/nnterp/rename_utils.py:49-51`; `TransformerLens/transformer_lens/model_bridge/supported_architectures/gemma4.py:3-24,52-55`; `TransformerLens/tests/integration/model_bridge/test_gemma4_bridge.py:1-100`; `llama.cpp/examples/eval-callback/eval-callback.cpp:53-77`; `TensorRT-LLM/examples/llm-api/quickstart_advanced.py:244-256,557-577`.

## Layer 6: Scheduling And Model Serving

This layer manages requests, KV-cache memory, batching, and a network API.

```text
[HTTP or Python request]
              |
              v
       [Request queue]
              |
              v
         [Scheduler] <----> [KV cache manager]
              |
              v
      [Inference workers]
              |
              v
          [Response]
              |
              v
[HTTP or Python response]
```

| Serving option | Behavior observed in cited range | Not established by cited range |
|---|---|---|
| vLLM | Request-queue types and KV-cache block data structures | Queue operation, full scheduling loop, prefix-cache behavior, and network routes |
| SGLang | Radix-tree node state and prefix-hash traversal | Cache lookup and update operations, request queue, scheduling loop, and network routes |
| llama-server | KV-cache slot and sequence operations | Request admission, cross-request prefix policy, and network routes |
| Ollama | Bounded pending channels and parallel loaded-model use | Slot scheduling, cross-request prefix policy, and route behavior |
| TensorRT-LLM | No Layer 6 source citation | Queue, scheduling, cache reuse, and network routes |
| TGI v3 | Token-budget queue decisions and requeue behavior | Complete prefix-cache and network-route behavior |

Use this layer for throughput studies. Do not use serving throughput as evidence that activation values are correct.

Project activity or archival status is repository metadata. Source code alone does not prove that status.

Local evidence: `vllm/vllm/v1/core/sched/scheduler.py:49-85`; `vllm/vllm/v1/core/kv_cache_manager.py:33-39`; `sglang/python/sglang/srt/mem_cache/radix_cache.py:250-294`; `llama.cpp/src/llama-kv-cache.cpp:747-894`; `ollama/server/sched.go:60-104,197-209`; `text-generation-inference/backends/v3/src/queue.rs:300-354`.

## Layer 7: Client, Gateway, And User Application

This layer sends requests to a model server. A client or gateway does not calculate model tensors. Some products in this layer also include a server and a lower inference engine.

```text
[Experiment or user]
          |
          +--> [Direct Python API]
          |
          +--> [OpenAI-compatible client]
          |
          +--> [LiteLLM gateway] --> [One or more model servers]
          |
          +--> [Ollama or LM Studio user interface]
```

### Client And Gateway Comparison

| Option | Role | Routes to many backends | API contract | Best use |
|---|---|---:|---|---|
| Direct runtime Python call | Client and local API | No | Runtime-specific | Controlled experiment |
| llama-cpp-python | Direct Python binding | No | Native Python binding | Python control of llama.cpp |
| OpenAI-compatible client | Client | No | Preserves the common OpenAI request shape | Behavioral evaluation |
| LiteLLM | Gateway | Not established by the cited example | One proxy example uses an OpenAI client shape | Candidate for one API across servers |

### Bundled Application Comparison

| Product | User interface | Model manager | Local server | Public activation API | Best use |
|---|---:|---:|---:|---:|---|
| Ollama | Server/API verified; user interface not assessed from routes | Yes | Yes | No activation endpoint found in inspected routes | Easy local model use |
| LM Studio | Not verified from local source | Not verified | Not verified | Not verified | Not assessed from the available local source set |

Absence claims apply only to the inspected public source snapshot. A private extension can add routes. LiteLLM can transport custom backend data if its integration supports that data.

Local evidence: `litellm/cookbook/litellm_proxy_server/mcp/mcp_with_litellm_proxy.py:1-15`; `vllm/examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py:27-39`; `ollama/server/routes.go:1823-1902`; `llama-cpp-python/llama_cpp/llama.py:55,592-666,1067,1815,2004`.

## Layer 8: Experiment Evidence

This layer records evidence required to assess reproducibility. It does not make a result reproducible by itself.

```text
[Artifact identity] ----+
[Environment versions] -+
[Serialized inputs] -----+--> [Experiment record]
[Activation definitions]-+
[Outputs and metrics] ---+
```

Evidence collection owner: the project experiment harness.

| Evidence group | Runtime support | Required experiment record |
|---|---|---|
| Artifact | Loaders can accept paths and revisions | Repository and resolved revisions, file manifest, checksums, license |
| Environment | Some tools report package, driver, CUDA, and GPU versions | Code commit, Nix and uv locks, OS, complete build provenance |
| Input | Tokenizers and benchmarks accept prompts, media, seeds, and templates | Raw messages, rendered template, token IDs, media hashes, tokenizer/config/processor revisions |
| Execution | Runtimes expose some dtype, context, cache, and scheduler settings | Complete normalized configuration, kernel path, seed scope, determinism status |
| Instrumentation | Research tools can save and replace selected values | Module path, token position, tensor shape/dtype, intervention definition |
| Output | Benchmarks report selected timing and throughput fields | Generated tokens, requested logits, peak VRAM, raw timing samples |
| Analysis | Runtime tools do not provide a full statistical record | Dataset revision, metric code commit, repetitions, raw metrics, uncertainty method |

No inspected runtime produces the complete record. Use an explicit value such as `not emitted by runtime; collected externally` when a field is unavailable.

Local evidence: `vllm/vllm/collect_env.py:33-70,418-435,597-607`; `sglang/python/sglang/benchmark/offline_throughput.py:37-99`; `llama.cpp/tools/llama-bench/llama-bench.cpp:132-141,219-240,323-458`; `nnsight/tests/test_tiny.py:22-83`.

## Cross-Layer Tool Profiles

### PyTorch With Transformers

This is the first reference candidate. It has the documented Gemma 4 PyTorch implementation.

Transformers supplies Gemma 4 model code. PyTorch runs the model. The source defines hidden-state outputs. The exact checkpoint and generation path still need a local `output_hidden_states=True` test. PyTorch forward hooks can read or replace module inputs and outputs.

Strengths:

- It uses the original Python model structure.
- It supports hidden states, hooks, interventions, and gradients.
- `Gemma4UnifiedProcessor` source supports text, image, video, and audio. Do not generalize that support to every Gemma 4 checkpoint.
- It is easy to compare code with the Google model example.
- It supports the hooks, gradients, and model outputs that activation studies use.

Weaknesses:

- It is not designed around vLLM-style continuous serving. Measure batch throughput before comparison.
- Saved activations can use much memory.
- Compiled or fused kernels can make some internal values hard to read.
- Quantized modules can have different names and behavior.

Dependencies:

- Python.
- PyTorch with CUDA support.
- Transformers.
- Accelerate for device placement when required.
- Safetensors for common weight files.
- The Gemma processor and tokenizer files.

### Unsloth With Transformers And PyTorch

Unsloth is a cross-layer optimization, loading, training, and export tool. It wraps and patches parts of the Transformers and PyTorch path. It is not a peer replacement for vLLM or llama.cpp.

Current source explicitly maps Gemma 4 E2B models, applies Gemma 4-specific loader behavior, and provides training paths such as LoRA, QLoRA, full fine-tuning, and preference or reinforcement-learning workflows. It also supplies custom PyTorch autograd functions and Triton kernels.

For activation research, start with eager execution and validate hooks at explicit PyTorch module boundaries. Native hooks and gradients remain possible in the PyTorch path, but fused kernels, compilation, quantization, and patched modules can change which intermediate values exist. Do not assume that an NNsight experiment has the same hook coverage or numerical output before and after Unsloth optimization.

If this project only downloads an Unsloth GGUF, it does not need the Unsloth Python package. The GGUF is an artifact and llama.cpp executes it. An Unsloth base E2B quantization is a candidate for the deferred base/non-IT path only if the project later permits community derivatives.

Local evidence: `unsloth/README.md:72-88,248-275,454-457`; `unsloth/unsloth/models/loader.py:124-141,1255-1259,1615-1621,2057-2074`; `unsloth/unsloth/models/_utils.py:1703-1709,3158-3169`; `unsloth/unsloth/kernels/cross_entropy_loss.py:288-380`.

### NNsight With Transformers

NNsight is a research interface on top of PyTorch models.

It can read and change internal model values. It can also do activation patching, steering, and gradient studies. It can run locally or through a remote NNsight service.

Strengths:

- It gives a concise activation research API.
- It can read and write module values.
- It supports gradients with the PyTorch backend.
- It uses the underlying model instead of a local inference server API.

Weaknesses:

- A new model architecture can need an adapter or direct validation.
- Tracing adds complexity.
- Saved values can use much CPU and GPU memory.

Dependencies:

- All PyTorch and Transformers dependencies.
- NNsight.

### TransformerLens

TransformerLens is a mechanistic interpretability library.

It gives named hook points and activation caches. This source snapshot has a Gemma 4 adapter and integration tests, including E2B/E4B structure handling.

Strengths:

- It has a research-focused activation API.
- It supports activation caches and interventions.
- Many interpretability examples use its concepts.
- It gives consistent names for common transformer components.

Weaknesses:

- We must validate numerical parity for the exact Gemma 4 E2B checkpoint.
- A bridge or converted model can differ from the original model path.
- New multimodal paths can have less coverage than the text path.
- Quantized model support can be less direct.

Dependencies:

- PyTorch.
- Transformers for model import paths.
- TransformerLens.

### nnterp

nnterp is a naming and workflow layer on top of NNsight. It tries to give common names to layers, attention modules, MLP modules, final normalization, and the output head.

Strengths:

- It uses the original underlying PyTorch model.
- It can reduce model-specific hook code.
- It supplies helpers for logit-lens, patching, and steering work.

Weaknesses:

- It is not a separate inference engine.
- It depends on NNsight.
- Gemma 4 support in nnterp still needs a direct mapping test.
- Direct query, key, and value helpers are not complete.
- Its multimodal support is not clear.

Use nnterp only after a local Gemma 4 compatibility and numerical parity test.

### vLLM

vLLM is an inference and serving engine that is designed for high-throughput workloads.

It uses PagedAttention and continuous batching. These methods improve GPU use when many requests run together.

vLLM supports many quantization formats. Google supplies Gemma 4 QAT weights in Compressed Tensors `w4a16` format for vLLM.

Strengths:

- It provides continuous scheduling, paged KV-cache management, and optimized kernels. Measure throughput on this computer.
- It has an offline Python API and a server API.
- It supports continuous batching.
- It supports several GPU quantization formats.

Weaknesses:

- It does not provide gradients in its normal inference path.
- Its custom kernels can produce results that differ from Transformers.
- Its GGUF support is experimental and under-optimized.
- Its normal public API does not expose all internal values.
- Some activation features need NNsight or a plugin.

Dependencies:

- Python.
- PyTorch.
- CUDA.
- Triton and runtime-specific kernels.
- vLLM.
- A supported model and quantization format.

### NNsight With vLLM

NNsight supplies a vLLM integration. This is not a native core vLLM activation API.

The backend can read and change module activations. It can cache activations for many prompts. It may retain some vLLM batching benefits; measure the exact workload.

Strengths:

- It joins activation access with high-throughput generation.
- It supports activation reads and interventions.
- It can collect values for many prompts.

Weaknesses:

- It does not support gradients.
- The documented path is text-oriented. One NNsight invoke maps to one request, although vLLM can batch multiple invokes.
- It enforces eager execution for hooks. This disables CUDA graphs and some compilation optimizations.
- Its tensors use a flat token layout.
- Numerical parity with the Transformers output is not established.
- Gemma 4 E2B needs a direct compatibility test.

Dependencies:

- All vLLM dependencies.
- NNsight.
- The NNsight vLLM support packages.

### llama.cpp

llama.cpp is a C and C++ inference engine.

It is a strong choice for GGUF files. It can offload model layers to CUDA. It can also run model parts on the CPU.

Strengths:

- It has strong GGUF and Q4 support.
- It has a small deployment stack.
- It works on many hardware types.
- It has a command-line tool and a server.
- It has an `eval-callback` debug example for tensor observation.

Weaknesses:

- The observed tensor interface is a low-level C++ debug example.
- It does not provide normal PyTorch hooks.
- It does not provide a normal exposed gradient path. The lower ggml library separately advertises automatic differentiation.
- The eval callback does not demonstrate activation replacement.
- Custom activation work can require a custom build.
- GGUF graph names can change between versions.

Dependencies:

- A C++ build or a packaged binary.
- CUDA build support for NVIDIA GPU use.
- A GGUF model file.

### Ollama

Ollama is a local model application and API service. It uses lower-level runtimes to run models.

Strengths:

- It is easy to install and run.
- It gives a local HTTP API.
- It manages model downloads and local storage.

Weaknesses:

- No activation or hidden-state endpoint was found in the inspected public server routes.
- It hides some lower-level runtime choices.
- Model tags can change unless the artifact is pinned.
- It is not the best base for activation research.

### LiteLLM

LiteLLM is not an inference engine.

It gives one API shape for many model services. It can route requests to Ollama, vLLM, cloud services, and other providers.

Use LiteLLM only if the project later needs one client API for multiple runtimes. It adds no activation access by itself. It can transport custom backend data only when an integration supports that data.

### TGI

TGI means Text Generation Inference. Hugging Face developed it as a model server.

[External repository metadata](https://github.com/huggingface/text-generation-inference) states that the upstream repository was archived on 2026-03-21. The inspected source does not prove repository status. Its published model list stops at Gemma 3 and does not list Gemma 4.

Do not select the inspected TGI snapshot for this Gemma 4 project unless a maintained, Gemma 4-compatible release is verified.

### SGLang

SGLang is an inference and serving system for high-throughput workloads. The cited source shows radix-tree cache structures. Request scheduling requires a separate source citation. SGLang is a candidate for agent work, structured output, and prompts that share a large prefix; measure the selected workload.

SGLang does not have a stable, high-level activation research API in the sources that we reviewed. Gemma 4 and Q4 support can change quickly. Test one exact release before use.

### TensorRT-LLM

TensorRT-LLM is an NVIDIA inference system. Its current source registers Gemma 4 model families and selected multimodal paths. Exact E2B execution still needs a local test.

It is designed for optimized NVIDIA deployment. It also has a large dependency set. This set includes the NVIDIA driver, CUDA, TensorRT, TensorRT-LLM, PyTorch, and often NVIDIA Model Optimizer.

Intermediate tensors usually need build-time changes. TensorRT-LLM is not a simple activation research system. Use it only after a benchmark shows a useful speed gain.

## Technical Terms

### Activation

An activation is a value that the model calculates during a forward pass. An activation is not usually a stored model weight.

### Hidden State

A hidden state is an activation that represents token data inside the model. Each transformer layer produces hidden states.

### Residual Stream

The residual stream is the main hidden-state path through transformer layers. Attention and MLP results add data to this path.

### Forward Pass

A forward pass is one model calculation from input data to output data.

### Hook

A hook is a function that runs when a selected model module runs. A hook can read or change module data.

### Intervention

An intervention changes an internal model value during a forward pass. Activation patching and activation steering are interventions.

### Gradient

A gradient shows how a result changes when an earlier value changes. Gradient research needs an automatic differentiation system such as PyTorch autograd.

### Weight

A weight is a learned model value. The runtime reads weights during inference.

### Model Artifact

A model artifact is a file or a set of files that stores model weights and related data. One trained model can have many artifacts with different formats or quantization methods.

### VRAM

VRAM is memory on a GPU. Model weights, activations, temporary data, and the KV cache use VRAM.

### Throughput

Throughput is the amount of work completed in a time period. LLM systems often report tokens per second.

### Latency

Latency is the time for one operation. Time to first token and time between output tokens are common LLM latency measurements.

### Quantization

Quantization stores or calculates values with fewer bits. It reduces memory use. It can also change model output.

### BF16

BF16 means bfloat16. It is a 16-bit floating-point format. It has a large numeric range and less precision than FP32.

### FP16

FP16 is a 16-bit floating-point format. It has more fraction precision than BF16 and a smaller numeric range.

### INT4 And Q4

INT4 uses 4-bit integer values. Q4 is a general name for 4-bit quantization. A Q4 method also needs scales, blocks, and other metadata.

### Q4_0

Q4_0 is one specific 4-bit block quantization method in the GGML and GGUF ecosystem. It is not the same as all other Q4 methods.

### PTQ

PTQ means Post-Training Quantization. PTQ quantizes a model after its main training is complete.

### QAT

QAT means Quantization-Aware Training. QAT simulates quantization during training so that the model can adjust to quantization errors.

### AWQ

AWQ means Activation-Aware Weight Quantization. It uses observed activations to decide how to protect important weights during weight quantization.

### GPTQ

GPTQ is a post-training weight quantization method. It uses approximate second-order information to reduce quantization error.

### `w4a16`

`w4a16` is a naming convention for 4-bit weights and 16-bit activations. Do not infer the selected checkpoint configuration from its name alone. Inspect its quantization configuration and ignored modules.

### `wNa8o8`

Google describes this as a mobile QAT schema. The cloned runtime source does not verify its package format or exact precision rules. Use the Google artifact documentation as the source.

### Safetensors

Safetensors is a tensor storage format. It stores model tensors without executable pickle code. A Transformers model usually also needs separate configuration and tokenizer files.

### Compressed Tensors

Compressed Tensors is a quantization schema and checkpoint convention layered on tensor files such as Safetensors. It stores quantization configuration with the checkpoint. vLLM can load supported configurations.

### KV Cache

KV means key and value. The KV cache stores attention keys and values from earlier tokens. It makes generation faster. It uses more memory when the context becomes longer.

### Kernel

A kernel is a compiled calculation that runs on a CPU, GPU, or other processor.

### Fused Kernel

A fused kernel joins multiple calculations into fewer kernels. It can improve speed. It can also prevent an intermediate value from being materialized for a hook.

### CUDA

CUDA is NVIDIA software for GPU calculation. PyTorch, vLLM, and llama.cpp can use CUDA.

### PagedAttention

PagedAttention is a vLLM memory method for KV cache data. It stores cache blocks in pages to reduce wasted GPU memory.

### Continuous Batching

Continuous batching adds and removes requests while the runtime is active. It improves throughput for many requests.

### Tokenizer

A tokenizer changes text into token numbers. A different tokenizer can change all model inputs.

### Chat Template

A chat template changes role messages into the exact token text that a chat model expects. A template change can change model output.

### Revision Pin

A revision pin selects one exact model repository commit. A branch name such as `main` is not an exact revision pin.

### Checksum

A checksum is a value calculated from file bytes. It can confirm that two model files are identical.

### Attention Weight

An attention weight shows how strongly one token position uses another token position in one attention calculation. A fused attention kernel can calculate the attention result without returning the full attention-weight matrix.

### Logit

A logit is an output score before the final probability calculation. The model has one logit for each possible next token.

### Eager Execution

Eager execution runs model operations as Python calls request them. It is easier to inspect than a compiled graph. It can be slower than compiled execution.

### Multimodal Model

A multimodal model accepts more than one input type. `Gemma4UnifiedProcessor` source supports text, image, video, and audio. The exact input paths for the selected Gemma 4 E2B checkpoint still require a local test. Each input path can have separate modules and activations.

### Deterministic Decoding

Deterministic decoding tries to produce the same output for the same input. A fixed seed and greedy decoding help. Different GPU kernels can still produce small numeric differences.

### Inference Engine

An inference engine performs the model forward pass and token generation. Transformers on PyTorch, vLLM, and llama.cpp can act as inference engines.

This document uses `runtime` only as an umbrella term for software that participates in model execution. The more specific terms are inference engine, model server, gateway, and application.

### Model Server

A model server gives a network API for an inference engine. vLLM, llama.cpp, Ollama, and TGI can act as model servers.

### Gateway

A gateway sends requests to one or more model servers. LiteLLM is a gateway. It does not calculate the model forward pass.

## Scientific Reproducibility Rules

A paper or experiment must record these items:

- The exact model name.
- The resolved model, tokenizer, configuration, processor, and projector revisions.
- The complete artifact manifest with file sizes and checksums.
- The quantization method.
- The quantization configuration, ignored modules, scales, group size, and cache precision.
- The inference engine name and version.
- The Python and package lock files.
- The CUDA and GPU driver versions.
- The GPU model.
- The raw messages, prompt text after chat-template processing, and final token IDs.
- The media file hashes and processed input tensor settings.
- The context size.
- All sampling settings.
- The random seed and its scope across Python, PyTorch, CUDA, sampler, and runtime components.
- The determinism status and observed repeatability results.
- The batch size.
- The request concurrency, prompt lengths, output lengths, and warm-up policy.
- The activation hook point names.
- The activation tensor shapes and data types.
- Whether the run used fused kernels or eager execution.
- The experiment code commit.
- The operating system version.
- The exact CUDA, cuDNN, and PyTorch builds.
- The model license and access conditions.
- The evaluation dataset name and revision.
- The metric definition and implementation.
- The number of repetitions.
- The raw metric output and repetition samples.
- Confidence intervals or another uncertainty measure.

The inference backend is an experiment variable. Do not assume that Transformers, vLLM, and llama.cpp give identical outputs.

## Recommended Test Plan

1. Load the official Gemma 4 E2B BF16 model with Transformers.
2. Run one fixed text prompt.
3. Save the residual stream from selected layers.
4. Change one selected activation with a PyTorch hook.
5. Repeat the test with NNsight.
6. Inspect the official QAT Compressed Tensors artifact configuration, then test it with vLLM.
7. Test activation capture with the NNsight vLLM backend.
8. Run the official QAT Q4_0 GGUF model with llama.cpp.
9. Compare tokens, speed, memory use, and output quality.
10. Record all artifact revisions and checksums.

### Required Adoption Gate

Do not adopt a stack until it passes all applicable checks below.

1. Pin the exact model artifact and revision.
2. Verify each file checksum.
3. Load the model from local files.
4. Run one text generation.
5. Capture one selected hidden state.
6. Change one selected activation.
7. Confirm that the intervention changes the expected metric.
8. Compare unmodified output with the official PyTorch reference.
9. Record peak VRAM and execution time.
10. Repeat the test after a clean environment build.

### Separate Acceptance Groups

The research baseline must reproduce the selected PyTorch model output and expose the required tensors.

The activation stack must capture and change named tensors at defined token positions.

The quantized deployment stack must meet a separate output-quality and memory target. Its activation values are not assumed to equal BF16 activation values.

The throughput stack must report warm-up policy, prompt lengths, output lengths, concurrency, time to first token, inter-token latency, and tokens per second.

## Current Recommendation

Use this order:

| Priority | Stack | Decision |
|---:|---|---|
| 1 | Transformers + PyTorch hooks | First candidate for the reference baseline |
| 2 | NNsight + Transformers | First candidate for the activation API |
| 3 | NNsight + vLLM | Test for large prompt sets |
| 4 | llama.cpp + official QAT Q4_0 GGUF | Use as a deployment reference |
| 5 | Ollama or LM Studio | Do not use for the main research path |
| 6 | LiteLLM | Defer until one API must route to many runtimes |

Do not select one quantized file as the only research artifact. Keep one high-precision model path and one deployment quantization path.

Use Unsloth only when measured training, memory, or export needs justify its patches. Keep the unmodified Transformers path as the reference baseline.

## Open Compatibility Tests

The research does not prove these items:

- NNsight operation with the exact Gemma 4 E2B revision.
- nnterp mapping for Gemma 4 E2B.
- Numerical parity between the implemented TransformerLens Gemma 4 adapter and the official model path.
- NNsight vLLM activation capture for Gemma 4 E2B.
- Multimodal activation capture with the NNsight vLLM backend.
- One stable SGLang release with the selected Gemma 4 QAT checkpoint.
- NNsight hook coverage and numerical parity after Unsloth Gemma 4 optimization.
- Provenance and quality of any selected Unsloth derivative quantization.

Run these tests before the project selects a final stack.

## Sources

### Local Source Snapshots

The following shallow clones are available under `/home/minttea/codebases/llms/`.

The commit hash identifies the source snapshot used for later code inspection. A shallow clone contains the selected commit and a limited history.

| Pipeline area | Project | Local path | Commit |
|---|---|---|---|
| Artifact format | ggml | `/home/minttea/codebases/llms/ggml` | `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` |
| Artifact format | compressed-tensors | `/home/minttea/codebases/llms/compressed-tensors` | `4feaf99fe86e62f99c123e42234ac9eaf88d51d5` |
| Model definition | Transformers | `/home/minttea/codebases/llms/transformers` | `a597f974857b3d92939971296bc0deb93d33d780` |
| Optimization and export | Unsloth | `/home/minttea/codebases/llms/unsloth` | `203007d19051dcd2ae33876786d117c99f6b0368` |
| Tensor execution | PyTorch | `/home/minttea/codebases/llms/pytorch` | `dcd2ecae775af66439b7ede4e7a82540b058c59c` |
| Tensor execution | Triton | `/home/minttea/codebases/llms/triton` | `3ca155e005df8807053faa04a194e67594ab46b2` |
| Tensor execution | llama.cpp | `/home/minttea/codebases/llms/llama.cpp` | `7e4c0a96880dae4fc4268ad441f8a6446bd5460a` |
| Tensor execution | vLLM | `/home/minttea/codebases/llms/vllm` | `c794754062d49a8fdb63ab3c5215b488b865030c` |
| Tensor execution | SGLang | `/home/minttea/codebases/llms/sglang` | `41cd5a718942f97bc45b0b5d7fca82992e8ae529` |
| Tensor execution | TensorRT-LLM | `/home/minttea/codebases/llms/TensorRT-LLM` | `0946d54b8d927433316f959f0ddd04ad570e0ba9` |
| Tensor execution | ExLlamaV2 | `/home/minttea/codebases/llms/exllamav2` | `7dc12af3a81f34ac3f27cd7602ed539b638933ca` |
| Instrumentation | NNsight | `/home/minttea/codebases/llms/nnsight` | `87f4dadaae254c6d36e8820cc83a251623b8dda7` |
| Instrumentation | TransformerLens | `/home/minttea/codebases/llms/TransformerLens` | `4194e13494c3369e8291c238d4943f821afcff25` |
| Instrumentation | nnterp | `/home/minttea/codebases/llms/nnterp` | `b4a31274692f986e493ac5d20ba20a4ec8640955` |
| Python binding | llama-cpp-python | `/home/minttea/codebases/llms/llama-cpp-python` | `67014fdf1d05ec2aae5bd574bf731182b884a03a` |
| Serving application | Ollama | `/home/minttea/codebases/llms/ollama` | `0f25c31bd53b64dc3fcc8fce0bde954159a67a58` |
| Gateway | LiteLLM | `/home/minttea/codebases/llms/litellm` | `f03df1bb42223ab8fa68033b01533295bf855188` |
| Serving stack | TGI | `/home/minttea/codebases/llms/text-generation-inference` | `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` |

LM Studio is not represented in this local source set. Its desktop product is not directly comparable to one repository in this set. Separate open-source projects can represent some underlying components.

### Online References

- Google Gemma 4 QAT model card: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf
- Google Gemma 4 unquantized QAT checkpoint: https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized
- Google Gemma 4 QAT Compressed Tensors model card: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- Google Gemma 4 instruction-tuned model card: https://huggingface.co/google/gemma-4-E2B-it
- Google Gemma 4 QAT announcement: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- Gemma 4 technical report: https://arxiv.org/abs/2607.02770
- TGI repository metadata: https://github.com/huggingface/text-generation-inference
- Transformers Gemma 4 documentation: https://huggingface.co/docs/transformers/main/en/model_doc/gemma4
- Transformers model output documentation: https://huggingface.co/docs/transformers/main_classes/output
- PyTorch hook documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
- NNsight overview: https://nnsight.net/about/
- NNsight vLLM support: https://nnsight.net/features/15_vllm_support/
- nnterp repository: https://github.com/ndif-team/nnterp
- TransformerLens architecture documentation: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
- vLLM GGUF documentation: https://docs.vllm.ai/en/latest/features/quantization/gguf/
- llama.cpp activation callback example: https://github.com/ggml-org/llama.cpp/tree/master/examples/eval-callback
- GGUF format specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Ollama API documentation: https://docs.ollama.com/api/introduction
- LiteLLM documentation: https://docs.litellm.ai/
- TensorRT-LLM Gemma examples: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma
- TGI archived repository: https://github.com/huggingface/text-generation-inference
- Unsloth repository: https://github.com/unslothai/unsloth
- ggml-org base Gemma 4 E2B GGUF: https://huggingface.co/ggml-org/gemma-4-E2B-GGUF/tree/0d85c394df5e9f3c5b07c791e31680e399042acc
