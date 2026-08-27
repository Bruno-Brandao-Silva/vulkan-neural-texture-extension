#!/usr/bin/env bash
set -e

MAX_ITERATIONS=5
ITERATION=1
APPROVED=false

echo "🚀 Iniciando Agent Loop Autônomo com Auditoria Visual Rigorosa..."

while [ $ITERATION -le $MAX_ITERATIONS ] && [ "$APPROVED" = false ]; do
  echo "--------------------------------------------------------"
  echo "🔄 Iteração $ITERATION de $MAX_ITERATIONS"
  echo "--------------------------------------------------------"

  # STEP 1: Agente Coder aplica ajustes no código
  echo "🤖 [Coder Agent] Aplicando alterações de código e layout no vntx-gui..."
  agy -p "Ajuste o layout do vntx-gui. Garanta containers width(Fill) e height(Fill) no app.rs e settings.rs. Remova max_width fixo. Se houver ícones quebrados ou caixas vazias '□', substitua por textos de rótulo limpos ou fontes nativas."

  # STEP 2: Agente Auditor de Código (Build + Formatação + Testes)
  echo "🧪 [Code Auditor] Validando testes e formatação..."
  if ./scripts/fmt_and_fix.sh && ./scripts/wsl2_test_runner.sh; then
    echo "✅ Código aprovado nos testes unitários e linter!"
  else
    echo "❌ Code Audit falhou. Notificando Coder Agent para correção..."
    ITERATION=$((ITERATION + 1))
    continue
  fi

  # STEP 3: Compilação e Captura Visual Headless (Xvfb + Imagemagick/Scrot)
  echo "📸 [Visual Auditor] Compilando e capturando screenshot da GUI..."
  cargo build --release -p vntx-gui
  rm -f /tmp/gui_audit.png

  # Inicia servidor virtual Xvfb (1920x1080), roda a GUI e tira o print da tela cheia (root window)
  xvfb-run -s "-screen 0 1920x1080x24" bash -c "
    ./target/release/vntx-gui &
    GUI_PID=\$!
    sleep 4
    scrot /tmp/gui_audit.png || import -window root /tmp/gui_audit.png
    kill \$GUI_PID 2>/dev/null || true
  "

  # Validação de segurança: o arquivo de imagem foi gerado?
  if [ ! -f /tmp/gui_audit.png ]; then
    echo "⚠️ Falha ao gerar /tmp/gui_audit.png. Repetindo iteração..."
    ITERATION=$((ITERATION + 1))
    continue
  fi

  # STEP 4: Validação Multimodal com Visão no agy
  echo "👁️ [Visual Auditor] Analisando captura /tmp/gui_audit.png..."
  AUDIT_RESULT=$(cat /tmp/gui_audit.png | agy -p "Análise visual de UI/UX desta GUI:
    1. O conteúdo está ocupando toda a largura da tela (1920px) sem espaço morto na direita?
    2. Existem caracteres ou glifos quebrados (caixas '□')?
    3. Os botões e formulários estão alinhados e fluidos?
    Responda estritamente 'APPROVED' na primeira linha se a tela estiver perfeita e fluida, ou detalhe as correções necessárias.")

  echo "📋 Parecer do Auditor Visual:"
  echo "$AUDIT_RESULT"

  if echo "$AUDIT_RESULT" | head -n 1 | grep -q "APPROVED"; then
    echo "🎉 [Visual Auditor] UI/UX Aprovada com Sucesso!"
    APPROVED=true
  else
    echo "⚠️ [Visual Auditor] Reprovado Visualmente. Realizando novo ciclo de ajustes..."
    ITERATION=$((ITERATION + 1))
  fi
done

if [ "$APPROVED" = true ]; then
  echo "✨ Loop concluído com sucesso. Gerando commit final..."
  git add .
  git commit -m "feat(gui): autonomous visual-audited UI/UX overhaul"
  git push origin main
else
  echo "🚨 O loop atingiu o limite de iterações sem aprovação visual completa."
  exit 1
fi