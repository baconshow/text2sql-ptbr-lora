# Checklist de publicação (Bacon-only)

> Este arquivo é o teu roteiro. NÃO faz parte do que recrutador lê. Pode deletar antes de publicar, ou manter — ele reforça que você é organizado.

## Tempo estimado total: ~75 min

### ⏱️ Bloco 1 — Publicar o código no GitHub (10 min)

```bash
# no teu computador, dentro da pasta text2sql-ptbr-lora/
git init
git add .
git commit -m "Initial commit: QLoRA fine-tune pipeline for PT-BR Text-to-SQL"

# Cria o repo vazio no GitHub primeiro: https://github.com/new
# Nome sugerido: text2sql-ptbr-lora
git remote add origin https://github.com/baconshow/text2sql-ptbr-lora.git
git branch -M main
git push -u origin main
```

### ⏱️ Bloco 2 — Rodar o notebook no Colab (40-50 min)

1. Abre https://colab.research.google.com
2. `File → Open notebook → GitHub tab → cola o link do repo`
3. `Runtime → Change runtime type → T4 GPU`
4. Na célula §1 do notebook, edita a URL do `git clone` pra apontar pro teu repo (já está em `https://github.com/baconshow/text2sql-ptbr-lora.git`, então só confirma).
5. `Runtime → Run all`
6. Na célula §9 (push pro HF Hub), ele vai pedir teu token. Cria um em https://huggingface.co/settings/tokens com **escopo Write** e cola.

O notebook faz: gera dataset → treina `base` (r=16) → treina `rank8` → avalia os dois → roda baseline sem fine-tune → plota loss curves → gera tabela de ablation → publica adapter no HF Hub.

### ⏱️ Bloco 3 — Atualizar o README com números reais (10 min)

Depois do notebook rodar:

1. No Colab, faz o download de:
   - `results/ablation_table.md`
   - `assets/loss_curves.png`
   - `runs/base/eval_results.json`
2. Copia e cola a tabela real do `ablation_table.md` em cima da tabela placeholder no `README.md` (seção **📈 Results**).
3. Substitui o `loss_curves.png` local pelo gerado no Colab.
4. Commit e push:

```bash
git add README.md assets/loss_curves.png results/ablation_table.md
git commit -m "Update README with actual training results"
git push
```

### ⏱️ Bloco 4 — Polimento final (10 min)

- [ ] Abrir teu repo no GitHub e checar se o SVG da arquitetura renderiza bonito no topo do README
- [ ] Confirmar que o **badge do Colab** abre o notebook direto (clica pra testar)
- [ ] Ir no teu perfil `github.com/baconshow`, na seção **Pinned repositories**, pinar o `text2sql-ptbr-lora`
- [ ] No repo, em `About` (roda de engrenagem no topo-direito): adicionar descrição curta + tópicos: `llm`, `fine-tuning`, `lora`, `qlora`, `peft`, `pytorch`, `huggingface`, `text-to-sql`, `portuguese`, `nlp`
- [ ] Confirmar que o adapter no HF Hub abre: `https://huggingface.co/baconshow/qwen2.5-0.5b-text2sql-ptbr-lora`

### ⏱️ Bloco 5 — Sinal forte no perfil (5 min)

No teu perfil GitHub (`baconshow/baconshow` README), adiciona uma seção tipo:

```markdown
### 🔬 LLM / Fine-tuning
- [**text2sql-ptbr-lora**](https://github.com/baconshow/text2sql-ptbr-lora) — QLoRA fine-tune of Qwen2.5-0.5B for Portuguese-to-SQL generation, with ablation study and synthetic data pipeline. [🤗 Model](https://huggingface.co/baconshow/qwen2.5-0.5b-text2sql-ptbr-lora)
```

---

## O que falar na aplicação (mini-pitch, 3 bullets)

- Built an end-to-end QLoRA fine-tuning pipeline (PyTorch + HF Transformers + PEFT + bitsandbytes) that adapts a 0.5B LLM to Portuguese text-to-SQL using synthetic data I designed — trained and published on Hugging Face Hub.
- Ran a multi-axis ablation study (LoRA rank ∈ {8, 16, 32}, target-module scope, learning rate) with exact-match / token-F1 / parse-validity metrics and per-template failure-mode analysis.
- Entire experiment is reproducible in ~45 min on a free Colab T4.
