import argparse
import os
import sys
import torch
from transformers import AutoTokenizer
import gradio as gr

# Ensure local src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.models.ultrathink import UltraThinkModel
from src.models.dynamic_reasoning import ReasoningPath, DynamicReasoningEngine


def load_model(model_dir: str, device: str = None, fallback_tokenizer: str = 'gpt2'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Try to load tokenizer from model_dir, else fallback to a known tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load custom model
    model = UltraThinkModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def format_chat_history_messages(history_msgs):
    # history_msgs is a list of {role, content}
    convo = ""
    for m in history_msgs:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        if role.lower() == 'system':
            # treat system as instruction
            convo += f"System: {content}\n"
        elif role.lower() == 'user':
            convo += f"User: {content}\n"
        else:
            convo += f"Assistant: {content}\n"
    return convo


def build_prompt(history_msgs, user_input):
    base = format_chat_history_messages(history_msgs)
    base += f"User: {user_input}\nAssistant:"
    return base


def _format_reasoning_md(trace):
    if not trace:
        return "No reasoning trace available."
    lines = ["### Reasoning trace"]
    # Compact timeline first
    timeline = []
    for step in trace:
        if isinstance(step, dict):
            timeline.append(str(step.get('chosen_path') or step.get('path') or 'unknown'))
        else:
            timeline.append('unknown')
    lines.append("**Timeline:** " + " → ".join(timeline))
    lines.append("")
    for i, step in enumerate(trace):
        if isinstance(step, dict):
            chosen = step.get('chosen_path') or step.get('path') or 'unknown'
            score = step.get('score') or step.get('confidence') or ''
            info = step.get('info') or ''
            lines.append(f"- Step {i+1}: path = `{chosen}`  score = `{score}`  {info}")
        else:
            lines.append(f"- Step {i+1}: {step}")
    return "\n".join(lines)


def generate_fn_builder(tokenizer, model, device):
    @torch.no_grad()
    def generate_fn(message, history_msgs, max_new_tokens, temperature, top_p, repetition_penalty, use_dre, reasoning_path_opt, enforce_safety, force_enable_dre, simulate_reasoning):
        prompt = build_prompt(history_msgs, message)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0.0,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Optionally attach an untrained DRE if requested and missing
        if bool(use_dre) and bool(force_enable_dre) and getattr(getattr(model, 'core', None), 'dre', None) is None:
            try:
                model.core.dre = DynamicReasoningEngine(
                    base_model=model.core.base_model,
                    config={'hidden_dim': model.config.model_config.n_embd}
                )
            except Exception:
                pass

        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            # Use custom model's generate method
            rp = None
            if reasoning_path_opt and reasoning_path_opt.lower() != 'auto':
                # Map string to ReasoningPath enum if possible
                try:
                    rp = getattr(ReasoningPath, reasoning_path_opt.upper())
                except Exception:
                    rp = None
            output_ids = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                use_dre=bool(use_dre),
                reasoning_path=rp,
                enforce_safety=bool(enforce_safety),
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only the last assistant segment after the final "Assistant:" marker
        if "Assistant:" in full_text:
            reply = full_text.split("Assistant:")[-1].strip()
        else:
            reply = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text.strip()
        # Retrieve reasoning trace if available
        reasoning = getattr(model, 'last_reasoning', None)
        if (not reasoning or len(reasoning) == 0) and bool(simulate_reasoning):
            # Produce a simple fake reasoning trace for demo purposes
            import random
            paths = ["fast", "standard", "deep", "ultra_deep"]
            steps = min(8, max(3, int(max_new_tokens) // 32))
            reasoning = [
                {"chosen_path": random.choice(paths), "confidence": round(random.uniform(0.5, 0.95), 2)}
                for _ in range(steps)
            ]
        reasoning_md = _format_reasoning_md(reasoning)
        return reply, reasoning, reasoning_md
    return generate_fn


def main():
    parser = argparse.ArgumentParser(description='Gradio UI for ULTRATHINK model')
    parser.add_argument('--model_dir', type=str, default=os.path.join('.', 'outputs', 'ultrathink_wikitext_best', 'final_model'))
    parser.add_argument('--server_name', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=7860)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_dir)
    generate_fn = generate_fn_builder(tokenizer, model, device)

    dre_available = bool(getattr(getattr(model, 'core', None), 'dre', None))
    with gr.Blocks(title="ULTRATHINK Chat") as demo:
        header = gr.Markdown("# ULTRATHINK Chat\nInteract with your trained model at: `" + args.model_dir + "`.")
        status_md = gr.Markdown(value=(
            "✅ DRE available: real reasoning trace will be shown."
            if dre_available else
            "⚠️ DRE not available in this checkpoint: showing placeholder reasoning (enable DRE and re-save to see real router decisions)."
        ))
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=500, type='messages')
                msg = gr.Textbox(placeholder="Type your message and press Enter...")
                with gr.Row():
                    clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(16, 512, value=128, step=8, label="Max new tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty")
                show_reasoning = gr.Checkbox(value=True, label="Show reasoning trace")
                use_dre = gr.Checkbox(value=True, label="Enable Dynamic Reasoning (DRE)")
                reasoning_path_opt = gr.Dropdown(
                    choices=["auto", "fast", "standard", "deep", "ultra_deep"],
                    value="auto",
                    label="Reasoning path"
                )
                enforce_safety = gr.Checkbox(value=False, label="Enforce safety during generation")
                with gr.Accordion("Reasoning trace", open=False):
                    reasoning_out = gr.JSON(label="last_reasoning (raw)")
                    reasoning_md = gr.Markdown(value="")
                force_enable_dre = gr.Checkbox(value=False, label="Force-enable DRE at runtime (untrained)")
                simulate_reasoning = gr.Checkbox(value=False, label="Simulate reasoning when DRE unavailable")

        def user_submit(user_message, chat_history_msgs):
            chat_history_msgs = (chat_history_msgs or []) + [{"role": "user", "content": user_message}]
            return "", chat_history_msgs

        def bot_respond(chat_history_msgs, max_new_tokens, temperature, top_p, repetition_penalty, show_reason, use_dre_v, reasoning_path_v, enforce_safety_v, force_dre_v, simulate_reason_v):
            # The last user message is at the end (messages format)
            last_user = chat_history_msgs[-1]["content"]
            reply, reasoning, reasoning_pretty = generate_fn(
                last_user,
                chat_history_msgs[:-1],
                max_new_tokens,
                temperature,
                top_p,
                repetition_penalty,
                use_dre_v,
                reasoning_path_v,
                enforce_safety_v,
                force_dre_v,
                simulate_reason_v,
            )
            chat_history_msgs = chat_history_msgs + [{"role": "assistant", "content": reply}]
            if show_reason:
                return chat_history_msgs, reasoning, reasoning_pretty
            else:
                return chat_history_msgs, gr.update(), gr.update()

        msg.submit(user_submit, [msg, chat], [msg, chat]).then(
            bot_respond,
            [chat, max_new_tokens, temperature, top_p, repetition_penalty, show_reasoning, use_dre, reasoning_path_opt, enforce_safety, force_enable_dre, simulate_reasoning],
            [chat, reasoning_out, reasoning_md]
        )
        clear_btn.click(lambda: ([], None, ""), None, [chat, reasoning_out, reasoning_md], queue=False)

    print(f"Launching ULTRATHINK Chat at http://{args.server_name}:{args.server_port}")
    demo.launch(server_name=args.server_name, server_port=args.server_port, show_api=False, inbrowser=True, share=False)


if __name__ == '__main__':
    main()
