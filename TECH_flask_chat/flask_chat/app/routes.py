from flask import Blueprint, render_template, request, session, jsonify
from datetime import datetime
from app import socketio
import os
from app.gemini.modelo import responder_pergunta

bp = Blueprint("chat", __name__)  # Blueprint para rotas do app

# === Funções auxiliares ===

def registrar_log(origem, mensagem, chat_id):
    os.makedirs("logs", exist_ok=True)
    caminho = f"logs/chat_{chat_id}.log"
    mensagem = mensagem.strip()
    if mensagem:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(caminho, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{origem}] {mensagem}\n")
        html = f"[{timestamp}] [{origem}] {mensagem}"
        socketio.emit("nova_mensagem", {"html": html})

def carregar_historico():
    chat_id = session.get("chat_id")
    caminho = f"logs/chat_{chat_id}.log"
    linhas_coloridas = []
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            linhas = list(f.readlines())
            for linha in linhas:
                if "[USUÁRIO]" in linha:
                    cor = "red"
                elif "[GEMINI]" in linha:
                    cor = "blue"
                elif "[JUIZ]" in linha:
                    cor = "purple"
                else:
                    cor = "black"
                linhas_coloridas.append(f'<font color="{cor}">{linha.strip()}</font>')
    return linhas_coloridas

# === Rotas de páginas ===

@bp.route('/')
def index():
    return render_template('pokemon.html')

@bp.route('/characters')
def characters():
    return render_template('characters.html')

@bp.route('/series')
def series():
    return render_template('series.html')

@bp.route('/filmes')
def filmes():
    return render_template('filmes.html')

@bp.route('/pokemon')
def pokemon():
    return render_template('pokemon.html')

@bp.route('/pessoas')
def pessoas():
    return render_template('pessoas.html')

# === Rota para processar pergunta ===

@bp.route('/pergunta', methods=['POST'])
def usuario():
    if "chat_id" not in session:
        session["chat_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        registrar_log("SISTEMA", f"=== Início da Sessão {session['chat_id']} ===", session["chat_id"])

    data = request.get_json()
    msg = data.get("mensagem", "")
    registrar_log("USUÁRIO", msg, session["chat_id"])

    resposta = {}
    if msg.strip().endswith("?"):
        resposta = responder_pergunta(msg)

        if isinstance(resposta, dict):
            registrar_log("GEMINI", resposta.get("resposta", "⚠️ Erro: resposta ausente"), session["chat_id"])
            registrar_log("JUIZ", resposta.get("avaliacao", "⚠️ Erro: avaliação ausente"), session["chat_id"])
        else:
            registrar_log("SISTEMA", "⚠️ Erro ao obter resposta do modelo", session["chat_id"])
            resposta = {"resposta": "Erro interno", "avaliacao": ""}

    return jsonify(resposta)
