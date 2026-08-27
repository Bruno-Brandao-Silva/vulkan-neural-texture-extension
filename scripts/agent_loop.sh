#!/usr/bin/env bash
set -e

MAX_ITERATIONS=5
ITERATION=1
APPROVED=false

echo "🚀 Iniciando Agent Loop Autônomo para Refatoração de GUI/UX..."

while [ $ITERATION -le $MAX_ITERATIONS ] && [ "$APPROVED" = false ]; do
  echo "--------------------------------------------------------"
  echo "🔄 Iteração $ITERATION de $MAX_ITERATIONS"
  echo "--------------------------------------------------------"

  # STEP 1: Agente Coder & Delegador executam a refatoração
  echo "🤖 [Coder Agent] Aplicando alterações de código e layout..."
agy -p "Ajuste o layout do vntx-gui. Garanta containers width(Fill) e height(Fill) no app.rs e settings.rs. Não use max_width fixo. Se houver falhas de build, corrija."

  # STEP 2: Agente Auditor de Código (Build + Formatação + Testes Headless)
  echo "🧪 [Code Auditor] Validando testes e formatação..."
  if ./scripts/fmt_and_fix.sh && ./scripts/wsl2_test_runner.sh; then
    echo "✅ Código aprovado nos testes unitários e linter!"
  else
    echo "❌ Code Audit falhou. Notificando Coder Agent para correção..."
    ITERATION=$((ITERATION + 1))
    continue
  fi

  # STEP 3: Agente Auditor Visual (Captura de Tela Headless no WSL2/Linux)
  echo "📸 [Visual Auditor] Compilando e capturando screenshot da GUI..."
  cargo build --release -p vntx-gui

  # Sobe o display virtual Xvfb, executa a GUI por 3 segundos e captura a tela
  xvfb-run -s "-screen 0 1920x1080x24" bash -c "
    ./target/release/vntx-gui &
    GUI_PID=\$!
    sleep 3
    scrot -u /tmp/gui_audit.png
    kill \$GUI_PID 2>/dev/null || true
  "

  # STEP 4: Validação Multimodal com o Agente de Visão
  echo "👁️ [Visual Auditor] Analisando layout da captura /tmp/gui_audit.png..."
  AUDIT_RESULT=$(cat /tmp/gui_audit.png | agy -p "Análise visual de UI/UX desta GUI:
1. O conteúdo está ocupando toda a largura da tela sem espaço morto excessivo na direita?
2. Existem caracteres ou glifos quebrados (caixas vazias '□')?
3. Os botões e formulários estão alinhados?
Responda estritamente 'APPROVED' na primeira linha se a tela estiver perfeita e fluida, ou detalhe as correções necessárias.")

  if echo "$AUDIT_RESULT" | grep -q "APPROVED"; then
    echo "🎉 [Visual Auditor] UI/UX Aprovada com Sucesso!"
    APPROVED=true
  else
    echo "⚠️ [Visual Auditor] Reprovado Visualmente. Feedbacks gerados:"
    echo "$AUDIT_RESULT"
    # Salva o feedback em um arquivo de contexto para a próxima iteração do agy
    echo "$AUDIT_RESULT" > /tmp/visual_feedback.txt
  fi

  ITERATION=$((ITERATION + 1))
done

if [ "$APPROVED" = true ]; then
  echo "✨ Loop concluído com sucesso. Gerando commit final..."
  git add .
  git commit -m "feat(gui): autonomous multi-agent UI/UX overhaul and visual polish"
  git push origin main
else
  echo "🚨 O loop atingiu o limite de iterações sem aprovação visual completa."
  exit 1
fi