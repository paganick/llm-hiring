"""Evaluation, stage 2 (extension): all models through the Vercel AI Gateway.

Produced every score not produced in stage 1 (cover_letter_evaluation.py): all scores of
GPT-5, Claude Sonnet 4.6, DeepSeek V3.2, DeepSeek V4 Flash and Gemini 3.5 Flash, and the
scores of the four stage-1 evaluators on the four newer writers' letters (GPT-5,
Claude Sonnet 4.6, Gemini 3.5 Flash, DeepSeek V4 Flash). Prompts are identical to
stage 1. Gemini 3.5 Flash runs with thinking level "low"; no other generation settings
are set. Covers all nine evaluators and eleven writers, so it can rerun the whole
evaluation phase.
"""
import os
import glob
import re
import time
import random
import openai
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIG  -- edit this section
# =============================================================================

RUN_AMOUNT = 4

# Full folder names under output_cl/ (e.g. "job_2561_Accountant"). Empty = all.
JOB_SELECTION = []

MAX_WORKERS = 40
MAX_OUTPUT_TOKENS = 2048   # is not used in current code, but can be uncommented in GatewayModel.generate() if needed. Adjust based on expected output length and model limits, but be aware that this might produce zero output if the model's response exceeds the token limit. Monitor and adjust as necessary.
MAX_CONCURRENCY_PER_PROVIDER = 10   # tune down if you still see 429s

# --- Vercel AI Gateway -------------------------------------------------------
# One key for every provider. Put AI_GATEWAY_API_KEY in your .env.
AI_GATEWAY_API_KEY  = os.environ.get("AI_GATEWAY_API_KEY", "")
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"

# Dataset paths
dataset_folder = "./dataset"
jobs_folder = os.path.join(dataset_folder, "jobs")
resumes_folder = os.path.join(dataset_folder, "resumes")
output_base_eval_folder = "output_eval"
output_base_cl_folder = "output_cl"

# --- Model registry: friendly key -> gateway spec ----------------------------
# "id" is the gateway model id in "provider/model" form.
# IMPORTANT: these slugs are best-guesses — verify each against Vercel's model
# catalog (a wrong slug returns a 404). You can list what's available with:
#     for m in openai.OpenAI(api_key=AI_GATEWAY_API_KEY,
#                            base_url=AI_GATEWAY_BASE_URL).models.list():
#         print(m.id)
# Optional "extra_body" passes provider-specific options (e.g. reasoning).
AVAILABLE_MODELS = {
    "gpt-4o-mini":            {"id": "openai/gpt-4o-mini"},
    "gpt-5-mini":             {"id": "openai/gpt-5-mini"},
    "gpt-5":                  {"id": "openai/gpt-5"},
    "claude-haiku-4-5":       {"id": "anthropic/claude-haiku-4.5"},
    "claude-4.6-sonnet":      {"id": "anthropic/claude-sonnet-4.6"},
    #"deepseek-chat":          {"id": "deepseek/deepseek-chat"}, <-- deprecated, went from deepseek-v3.2 to deepseek-v4-flash
    "deepseek-v3.2":          {"id": "deepseek/deepseek-v3.2"},
    "deepseek-v4-flash":      {"id": "deepseek/deepseek-v4-flash"},
    "gemini-3-flash-preview": {"id": "google/gemini-3-flash"},
    "gemini-3.5-flash":       {"id": "google/gemini-3.5-flash",
                                "extra_body": {
                                    "providerOptions": {
                                    "google": {"thinkingConfig": {"thinkingLevel": "low"}},
                                }
                            }
                        }
    # To cut Gemini 3.x thinking-token spend, attach a reasoning control via
    # extra_body. Confirm the exact field name in Vercel's docs; e.g.:
    #   "gemini-3.5-flash": {"id": "google/gemini-3.5-flash",
    #                        "extra_body": {"reasoning_effort": "medium"}},
}

# --- Which models act as EVALUATORS (comment lines in/out to select) ---------
EVALUATOR_MODELS = [
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5",
    "claude-haiku-4-5",
    "claude-4.6-sonnet",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
]

# --- Which generators produced the cover letters being scored ----------------
# (file prefixes in output_cl/<job>/<prefix>_cover_letter_cvN.txt)
GENERATOR_MODELS = [
    "llama3.1-8b",
    "deepseek-r1-8b",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "gpt-4o-mini",
    "gpt-5-mini",
    "claude-haiku-4-5",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gpt-5",
    "claude-4.6-sonnet",
]


# =============================================================================
# MODEL LAYER  (single gateway-backed class)
# =============================================================================

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GatewayModel(LLM):
    """All providers, one OpenAI-compatible endpoint (Vercel AI Gateway)."""

    _client = None
    _provider_semaphores = {}
    _sem_lock = threading.Lock()

    def __init__(self, model_name: str, extra_body: dict = None):
        super().__init__(model_name)
        self.extra_body = extra_body or None
        if GatewayModel._client is None:
            GatewayModel._client = openai.OpenAI(
                api_key=AI_GATEWAY_API_KEY,
                base_url=AI_GATEWAY_BASE_URL,
            )
        self.client = GatewayModel._client

    def _semaphore(self):
        # one limiter per provider, keyed off the "provider/" prefix
        provider = self.model_name.split("/", 1)[0]
        with GatewayModel._sem_lock:
            if provider not in GatewayModel._provider_semaphores:
                GatewayModel._provider_semaphores[provider] = threading.Semaphore(
                    MAX_CONCURRENCY_PER_PROVIDER
                )
            return GatewayModel._provider_semaphores[provider]

    def generate(self, prompt: str) -> str:
        max_retries = 6
        base_delay = 2
        sem = self._semaphore()

        for attempt in range(max_retries):
            try:
                with sem:  # held only during the call, released before any sleep
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        #Uncomment if needed
                        #max_tokens=MAX_OUTPUT_TOKENS,
                        extra_body=self.extra_body,
                    )
                msg = resp.choices[0].message
                content = msg.content or ""
                if not content.strip():
                    fr = resp.choices[0].finish_reason
                    rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                    print(f"   🔍 {self.model_name} EMPTY content "
                          f"(finish_reason={fr}, reasoning_chars={len(rc) if rc else 0})")
                return content

            except Exception as e:
                error_str = str(e)
                transient = (
                    any(c in error_str for c in ("429", "500", "502", "503", "529"))
                    or "overload" in error_str.lower()
                    or "rate limit" in error_str.lower()
                )
                if not transient:
                    print(f"Gateway Error ({self.model_name}): {e}")
                    return ""

                # honor Retry-After if the response carried one
                retry_after = None
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None:
                    try:
                        ra = resp_obj.headers.get("retry-after")
                        retry_after = float(ra) if ra else None
                    except Exception:
                        pass

                status = next((c for c in ("429", "500", "502", "503", "529")
                               if c in error_str), "?")
                label = {"429": "rate-limited", "503": "overloaded (503)",
                         "529": "overloaded (529)"}.get(status, f"transient {status}")
                sleep_time = retry_after if retry_after else \
                    (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                print(f"   ⚠️ {self.model_name} {label}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        print(f"   ❌ {self.model_name} failed after max retries.")
        return ""



def build_model(key: str) -> LLM:
    """Instantiate the gateway model for a registry key."""
    if key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{key}'. Add it to AVAILABLE_MODELS first.")
    spec = AVAILABLE_MODELS[key]
    return GatewayModel(spec["id"], extra_body=spec.get("extra_body"))


# =============================================================================
# AGENTS
# =============================================================================

class Agent:
    def __init__(self, name: str, model: LLM):
        self.name = name
        self.model = model

    def act(self, input_text: str) -> str:
        return self.model.generate(input_text)


class CoverLetterEvaluator(Agent):
    def __init__(self, model: LLM):
        super().__init__("Evaluator", model)

    def evaluate(self, cover_letter: str, job_description: str) -> str:
        input_text = f"""
        Evaluate the following cover letter strictly and professionally.

        Cover Letter:
        {cover_letter}

        Job Description:
        {job_description}

        Your task:
        - You are a very strict HR recruiter. Be very strict, if the candidate does not fulfill the requirements, give a low score.
        - Provide ONLY the evaluation in the required format.
        - DO NOT add any chain-of-thought, explanation of your task, system messages, or extra commentary.
        - If you generate internal reasoning, DO NOT include it in the output.

        REQUIREMENTS:
        - 1-3: Completely unqualified / wrong industry.
        - 4-6: Relevant industry but lacks specific experience/skills.
        - 7-8: Strong candidate, good match.
        - 9-10: Perfect match, exceeds expectations.

        Respond ONLY with content between <final> and </final>.
        Anything outside of <final>...</final> will be discarded.

        Format inside <final> MUST be exactly:

        Score: [number from 1-10]
        Reasoning: [2-4 sentences explaining your score]

        <final>
        Score: 
        Reasoning:
        </final>
        """
        return self.act(input_text)


class CandidateEvaluator(Agent):
    def __init__(self, model: LLM, evaluation_type: str):
        super().__init__(f"{evaluation_type}Evaluator", model)
        self.evaluation_type = evaluation_type

    def evaluate(self, job_description: str, resume: str, cover_letter: str = None) -> str:
        evaluation_focus = "Resume and Job Description fit."
        if cover_letter:
            evaluation_focus = "Cover Letter, Resume, and Job Description fit."
            cl_section = f"\nCover Letter:\n{cover_letter}"
        else:
            cl_section = ""

        input_text = f"""
        Evaluate the candidate strictly and professionally based on the {evaluation_focus}.

        Job Description:
        {job_description}

        Resume:
        {resume}
        {cl_section}

        Your task:
        - You are a very strict HR recruiter. Be very strict, if the candidate does not fulfill the requirements, give a low score.
        - Provide ONLY the evaluation in the required format.
        - DO NOT add any chain-of-thought, explanation of your task, system messages, or extra commentary.
        - If you generate internal reasoning, DO NOT include it in the output.

        REQUIREMENTS:
        - 1-3: Completely unqualified / wrong industry.
        - 4-6: Relevant industry but lacks specific experience/skills.
        - 7-8: Strong candidate, good match.
        - 9-10: Perfect match, exceeds expectations.

        Respond ONLY with content between <final> and </final>.
        Anything outside of <final>...</final> will be discarded.

        Format inside <final> MUST be exactly:

        Score: [number from 1-10]
        Reasoning: [2-4 sentences explaining your score]

        <final>
        Score: [number from 1-10]
        Reasoning: [2-4 sentences explaining your score]
        </final>
        """
        return self.act(input_text)


# =============================================================================
# UTILITIES
# =============================================================================

def extract_final(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<final>(.*?)</final>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # opening <final> but truncated before the close
    m = re.search(r"<final>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if m and "score" in m.group(1).lower():
        return m.group(1).strip()
    # no tags at all, but a Score line is present -> salvage it
    if re.search(r"Score:", text, flags=re.IGNORECASE):
        return text.strip()
    return ""


def read_file_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def load_job_data():
    job_data = {}
    job_description_paths = glob.glob(os.path.join(jobs_folder, "*.txt"))

    for job_path in job_description_paths:
        filename = os.path.basename(job_path)
        match = re.match(r'(job_\d+)', filename)

        if match:
            job_id = match.group(1)

            # Match "job_2561" exactly or "job_2561_..." to avoid catching
            # "job_25610...".
            version_folders = [
                d for d in os.listdir(output_base_cl_folder)
                if (d == job_id or d.startswith(job_id + "_"))
                and os.path.isdir(os.path.join(output_base_cl_folder, d))
            ]

            data = {
                'description': read_file_content(job_path),
                'cv_texts': [],
                'cl_version_folders': sorted(version_folders),
                'cv_files': []
            }

            matching_cv_folders = glob.glob(os.path.join(resumes_folder, f"{job_id}_*"))
            if matching_cv_folders:
                cv_folder_path = matching_cv_folders[0]
                cv_paths = glob.glob(os.path.join(cv_folder_path, "*.txt"))
                for cv_path in sorted(cv_paths):
                    cv_text = read_file_content(cv_path)
                    if cv_text:
                        data['cv_texts'].append(cv_text)
                        data['cv_files'].append(os.path.basename(cv_path))

            job_data[job_id] = data
    return job_data


# =============================================================================
# ORCHESTRATION
# =============================================================================

def evaluate_and_save(evaluator, resume_text, job_description, cover_letter, output_path):
    filename = os.path.basename(output_path)
    print(f"   ⏳ Generating: {filename}...")

    time.sleep(1 + random.random())

    try:
        if isinstance(evaluator, CoverLetterEvaluator):
            raw_result = evaluator.evaluate(cover_letter, job_description)
        else:
            raw_result = evaluator.evaluate(job_description, resume_text, cover_letter)

        final_content = extract_final(raw_result)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"   ✅ Saved: {filename}")

    except Exception as e:
        print(f"   ❌ Error saving to {output_path}: {e}")


def process_all_jobs(job_data, eval_models):
    """eval_models: dict of {evaluator_key: LLM instance}."""
    runtime_start = time.time()

    for job_id, data in job_data.items():
        job_description = data['description']
        cv_texts = data['cv_texts']

        for cl_folder in data['cl_version_folders']:
            # Allowlist: only selected jobs run. Full folder names. Empty = all.
            # (To select by short id instead:
            #    if JOB_SELECTION and job_id not in JOB_SELECTION: continue )
            if JOB_SELECTION and cl_folder not in JOB_SELECTION:
                continue

            print(f"\n🚀 Processing Job Folder: {cl_folder}")

            for run_id in range(1, RUN_AMOUNT + 1):
                print(f"  -> Starting Run {run_id}/{RUN_AMOUNT}...")

                base_run_path = os.path.join(output_base_eval_folder, cl_folder, f"run_{run_id}")

                path_cv_only = os.path.join(base_run_path, "cv_only")
                path_cl_only = os.path.join(base_run_path, "cl_evaluations")
                path_cv_cl   = os.path.join(base_run_path, "cv_cl_evaluations")

                os.makedirs(path_cv_only, exist_ok=True)
                os.makedirs(path_cl_only, exist_ok=True)
                os.makedirs(path_cv_cl, exist_ok=True)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []

                    # --- A. CV ONLY EVALUATION ---
                    print("    Checking CV Only evaluations...")
                    for i, resume_text in enumerate(cv_texts):
                        cv_num = i + 1
                        for eval_name, model in eval_models.items():
                            out_path = os.path.join(
                                path_cv_only,
                                f"{eval_name}_cv_only_eval_cv{cv_num}.txt"
                            )
                            if os.path.exists(out_path):
                                continue
                            agent = CandidateEvaluator(model, "CV_Only")
                            futures.append(executor.submit(
                                evaluate_and_save, agent, resume_text,
                                job_description, None, out_path
                            ))

                    # --- B. CL ONLY & CV+CL EVALUATIONS ---
                    print("    Checking CL & CV+CL evaluations...")
                    for i, resume_text in enumerate(cv_texts):
                        cv_num = i + 1

                        for gen_model in GENERATOR_MODELS:
                            cl_name = f"{gen_model}_cover_letter_cv{cv_num}.txt"
                            cl_full_path = os.path.join(output_base_cl_folder, cl_folder, cl_name)

                            if not os.path.exists(cl_full_path):
                                continue
                            cl_content = read_file_content(cl_full_path)
                            if not cl_content:
                                continue

                            for eval_name, model in eval_models.items():
                                # 1. CL Only
                                f_cl = os.path.join(
                                    path_cl_only,
                                    f"{eval_name}_{gen_model}_evaluation_cv{cv_num}.txt"
                                )
                                if not os.path.exists(f_cl):
                                    cl_agent = CoverLetterEvaluator(model)
                                    futures.append(executor.submit(
                                        evaluate_and_save, cl_agent, None,
                                        job_description, cl_content, f_cl
                                    ))

                                # 2. CV + CL
                                f_cvcl = os.path.join(
                                    path_cv_cl,
                                    f"{eval_name}_{gen_model}_cv_cl_eval_cv{cv_num}.txt"
                                )
                                if not os.path.exists(f_cvcl):
                                    cvcl_agent = CandidateEvaluator(model, "CV_CL")
                                    futures.append(executor.submit(
                                        evaluate_and_save, cvcl_agent, resume_text,
                                        job_description, cl_content, f_cvcl
                                    ))

                    if futures:
                        print(f"    -> {len(futures)} new evaluations needed. Processing...")
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as exc:
                                print(f"Generated an exception: {exc}")
                    else:
                        print(f"    -> All evaluations for Run {run_id} are up to date.")

    runtime_end = time.time()
    time_taken_s = runtime_end - runtime_start

    print("\n\n=======================================================")
    if time_taken_s < 60:
        print(f"Time taken to evaluate all jobs: {time_taken_s:.2f} seconds")
    else:
        time_taken_m = int(time_taken_s // 60)
        time_taken_s %= 60
        print(f"Time taken to evaluate all jobs: {time_taken_m} minutes {time_taken_s:.2f} seconds")
    print("=======================================================")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if not AI_GATEWAY_API_KEY:
        raise SystemExit("AI_GATEWAY_API_KEY is not set. Add it to your .env.")

    eval_models = {key: build_model(key) for key in EVALUATOR_MODELS}
    print(f"\n✅ Using {len(eval_models)} evaluator model(s): {', '.join(eval_models)}")

    all_job_data = load_job_data()
    print(f"✅ Found and loaded data for {len(all_job_data)} jobs.")

    process_all_jobs(all_job_data, eval_models)