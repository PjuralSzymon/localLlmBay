# Licence and legal summary (LocalLLMBay)

This file is the **general licence for LocalLLMBay project source that we publish** (today: the contribution **node** in this folder). Using the **hosted service** (https://localllmbay.com) is a separate contract: Terms of Service, Privacy, Cookies, and Legal notice on the website.

This is a summary for operators and people reading public node code. It is **not legal advice** and does not replace those website documents or any upstream model card.

**Governing law for the service:** Poland, except where mandatory EU or consumer rules say otherwise.  
**Product policy:** the beta is intended for people **18 or older**.

---

## 1. LocalLLMBay source code (MIT)

Copyright (c) 2026 LocalLLMBay contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**What this covers:** our Python/shell/Docker sources in this `node/` tree (and, when we publish them, other LocalLLMBay originals under the same terms).

**What this does not cover:** model **weights**, the **Ollama** binary inside the image, **PyTorch / Diffusers / CUDA**, fonts, third-party names, or your copy of someone else’s prompt. Those keep their own licences. MIT on our glue code does **not** re-licence Granite, Qwen, Flux, Wan, Gemma, or similar.

The website, API, and queue source are **not all public** at the moment. When they are, the same MIT grant is intended to apply to **our** code in those trees unless a file says otherwise.

---

## 2. Hosted service (not this MIT file)

If you create an account, run a node against the public API, or send jobs:

- You agree to the **Terms of Service** on the site (`localllmbay-tos`).
- Privacy and cookies are described in the **Privacy** and **Cookies** notices.
- The **Legal notice** must show the operator’s name, address, and email when the service is offered to the public. If those fields are empty, the service is not offered to the public.

Beta: experimental, no SLA, community machines, FIFO matching. Output is AI-generated. Do not rely on it as sole production infrastructure.

---

## 3. What running a node legally implies (plain language)

You install **official** `localllmbay-node` software and pull jobs for other people.

- A matched job is processed in **readable form** on your computer (Ollama / Diffusers). Encoding on the wire is **not** secrecy against you.
- Terms **forbid** recording, training on, publishing, or selling job content. Technical ability to see a prompt is not permission.
- This is **not** end-to-end confidential inference. Do not tell users that it is.
- Tools (`write`, `bash`, `read`, …) run on the **requester’s** PC, not on the node.
- Do not submit (and do not invite others to submit) passwords, private keys, health records, or other data that cannot sit on an unknown volunteer PC.

Default matching aims at **EEA / EU adequacy** locations; classification can be wrong. GDPR roles of node operators are for counsel — the Terms still require confidentiality of job content.

---

## 4. Third-party runtimes and model weights

LocalLLMBay chooses an allowlist and the official image **pulls** weights onto the operator disk. Confirm the **exact tag or Hugging Face repo** at each bump. This table is an inventory, not a substitute for the upstream text.

| Piece | Typical licence (verify) | Notes |
|-------|-------------------------|--------|
| Ollama in the image | MIT of that Ollama version | Runtime for text; not our product name |
| “Ollama” name / logo | Trademark | Factual attribution only |
| PyTorch, Diffusers, CUDA wheels | Their upstream licences | Image/video path |
| IBM Granite 3.3 8B (`granite3.3:8b`) | Apache-2.0 on the Granite 3.3 card | Graph orchestrator |
| Granite 4.2 leftover ladder | Confirm the card | Leftover text family |
| DeepSeek-Coder 6.7B | Confirm the Ollama card | Graph thinking |
| Qwen 2.5 Coder | Confirm (Apache-2.0 / Tongyi variants exist) | Graph final |
| Qwen 3.5 / Ministral / DeepSeek-R1 distills | Confirm each card | Leftover ladders |
| Gemma 3 4B | Confirm Gemma / Google card | Docs graph |
| FLUX.2 Klein 4B (`black-forest-labs/FLUX.2-klein-4B`) | Apache-2.0 on that card | Diffusers; **not** the gated 9B non-commercial Klein |
| Wan 2.1 T2V 1.3B (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) | Apache-2.0 on that card | Diffusers video |

Redistributing a built Docker image may distribute **those** components too. Follow their terms, notices, and any use restrictions (including acceptable-use rules on image/video models).

---

## 5. Trademarks and affiliation

**LocalLLMBay** is the product name. We are **not** affiliated with Ollama, IBM, Alibaba/Qwen, Google, Black Forest Labs, Wan-AI, Mistral, DeepSeek, Hugging Face, NVIDIA, or Microsoft. Mentions are for identification of software or models the node can run.

---

## 6. Warranty and liability (code + beta)

The MIT text above disclaims warranty for the **published source**. The hosted beta also disclaims SLA and production reliance, **subject to mandatory law** (including EU consumer and GDPR rights that cannot be waived).

Generated code, images, and video can be wrong, unsafe, or infringing. Review before you use them.

---

## 7. Comments

Licence or attribution mistakes (wrong model card, missing notice): open a GitHub issue on this repo. See `README.md` in this folder.
