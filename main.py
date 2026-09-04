import discord
from discord.ext import commands

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

import os
import random
import logging
import asyncio


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TOKEN = "seu_token"

PREFIX = "!"


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents
)


# ============================================================
# LOG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("jogo")


# ============================================================
# CARREGAR CLIP
# ============================================================

logger.info("Carregando modelo CLIP...")

try:

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    logger.info("CLIP carregado com sucesso!")

except Exception as erro:

    logger.exception(
        f"Erro ao carregar CLIP: {erro}"
    )

    model = None
    processor = None


# ============================================================
# JOGOS ATIVOS
# ============================================================

# Guarda o objeto escolhido por canal.
#
# Exemplo:
#
# jogos[123456] = "celular"

jogos = {}


# ============================================================
# OBJETOS DO JOGO
# ============================================================

# Cada objeto possui algumas descrições.
#
# Isso é importante porque CLIP entende linguagem.
#
# Por exemplo:
#
# celular:
#   smartphone
#   cell phone
#   mobile phone
#
# copo:
#   drinking glass
#   glass cup
#   cup
#
# Isso aumenta bastante a chance de reconhecimento.

OBJETOS = {

    "celular": [
        "a smartphone",
        "a cell phone",
        "a mobile phone",
        "a smartphone device"
    ],

    "copo": [
        "a drinking glass",
        "a glass cup",
        "a cup",
        "a glass for drinking"
    ],

    "mouse": [
        "a computer mouse",
        "a PC mouse",
        "a wireless mouse"
    ],

    "livro": [
        "a book",
        "a printed book",
        "a book on a table"
    ],

    "garrafa": [
        "a bottle",
        "a drinking bottle",
        "a water bottle"
    ],

    "monitor": [
        "a computer monitor",
        "a computer screen",
        "a desktop monitor"
    ],

    "teclado": [
        "a computer keyboard",
        "a PC keyboard",
        "a keyboard"
    ],

    "fone": [
        "headphones",
        "earphones",
        "a pair of headphones",
        "a headset"
    ],

    "oculos": [
        "eyeglasses",
        "a pair of glasses",
        "reading glasses"
    ],

    "relogio": [
        "a wristwatch",
        "a watch",
        "a wrist watch"
    ],

    "mochila": [
        "a backpack",
        "a school backpack",
        "a bag backpack"
    ],

    "caneta": [
        "a pen",
        "a ballpoint pen",
        "a writing pen"
    ],

    "tenis": [
        "a sneaker",
        "a pair of sneakers",
        "a running shoe"
    ],

    "maca": [
        "an apple",
        "a red apple",
        "a fruit apple"
    ]

}


# ============================================================
# NOMES BONITOS PARA O DISCORD
# ============================================================

NOMES = {

    "celular": "📱 celular",
    "copo": "🥛 copo",
    "mouse": "🖱️ mouse",
    "livro": "📚 livro",
    "garrafa": "🍾 garrafa",
    "monitor": "🖥️ monitor",
    "teclado": "⌨️ teclado",
    "fone": "🎧 fone",
    "oculos": "👓 óculos",
    "relogio": "⌚ relógio",
    "mochila": "🎒 mochila",
    "caneta": "🖊️ caneta",
    "tenis": "👟 tênis",
    "maca": "🍎 maçã"

}


# ============================================================
# FUNÇÃO DE ANÁLISE
# ============================================================

def analisar_imagem(caminho, objeto_escolhido):
    """
    Analisa a imagem usando CLIP.

    Retorna:
        melhor_objeto
        confianca
        resultados
    """

    if model is None or processor is None:
        raise RuntimeError(
            "O modelo CLIP não está carregado."
        )

    # --------------------------------------------------------
    # ABRIR IMAGEM
    # --------------------------------------------------------

    imagem = Image.open(
        caminho
    ).convert("RGB")


    # --------------------------------------------------------
    # CRIAR CANDIDATOS
    # --------------------------------------------------------

    candidatos = list(
        OBJETOS.keys()
    )


    # --------------------------------------------------------
    # GERAR TEXTOS
    # --------------------------------------------------------

    textos = []

    mapa_textos = []

    for objeto in candidatos:

        descricoes = OBJETOS[objeto]

        for descricao in descricoes:

            textos.append(
                f"a photo of {descricao}"
            )

            mapa_textos.append(
                objeto
            )


    # --------------------------------------------------------
    # PROCESSAR
    # --------------------------------------------------------

    inputs = processor(
        text=textos,
        images=imagem,
        return_tensors="pt",
        padding=True
    )


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    # --------------------------------------------------------
    # PEGAR LOGITS
    # --------------------------------------------------------

    logits = outputs.logits_per_image[
        0
    ]


    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    probabilidades = torch.softmax(
        logits,
        dim=0
    )


    # --------------------------------------------------------
    # SOMAR AS VARIAÇÕES DE CADA OBJETO
    # --------------------------------------------------------

    pontuacoes = {}

    for i, objeto in enumerate(
        mapa_textos
    ):

        valor = float(
            probabilidades[i]
        )

        if objeto not in pontuacoes:
            pontuacoes[objeto] = []

        pontuacoes[objeto].append(
            valor
        )


    # --------------------------------------------------------
    # MÉDIA
    # --------------------------------------------------------

    medias = {}

    for objeto, valores in pontuacoes.items():

        medias[objeto] = sum(
            valores
        ) / len(valores)


    # --------------------------------------------------------
    # NORMALIZAR ENTRE OS OBJETOS
    # --------------------------------------------------------

    total = sum(
        medias.values()
    )

    if total > 0:

        medias = {
            objeto: valor / total
            for objeto, valor in medias.items()
        }


    # --------------------------------------------------------
    # ORDENAR
    # --------------------------------------------------------

    resultados = sorted(
        medias.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # MELHOR RESULTADO
    # --------------------------------------------------------

    melhor_objeto = resultados[0][0]

    melhor_confianca = resultados[0][1]


    return (
        melhor_objeto,
        melhor_confianca,
        resultados
    )


# ============================================================
# BOT ONLINE
# ============================================================

@bot.event
async def on_ready():

    print()
    print("======================================")
    print("       BOT ONLINE!")
    print("======================================")
    print(f"Bot: {bot.user}")
    print(f"ID: {bot.user.id}")
    print("======================================")
    print()


# ============================================================
# PING
# ============================================================

@bot.command()
async def ping(ctx):

    await ctx.send(
        "🏓 Pong!"
    )


# ============================================================
# HELLO
# ============================================================

@bot.command()
async def hello(ctx):

    await ctx.send(
        "Olá! 👋"
    )


# ============================================================
# AJUDA
# ============================================================

@bot.command()
async def objeto(ctx):

    await ctx.send(
        "🎮 **COMO JOGAR**\n\n"

        "1️⃣ Digite `!jogodoobjeto`\n\n"

        "2️⃣ Eu escolherei um objeto.\n\n"

        "3️⃣ Tire ou envie uma foto.\n\n"

        "4️⃣ Mande a foto junto com `!adivinhar`.\n\n"

        "5️⃣ Minha IA vai analisar a imagem.\n\n"

        "🏆 Se a imagem corresponder ao objeto, "
        "você ganha!"
    )


# ============================================================
# COMEÇAR JOGO
# ============================================================

@bot.command()
async def jogodoobjeto(ctx):

    # Escolher objeto
    objeto_escolhido = random.choice(
        list(OBJETOS.keys())
    )


    # Salvar jogo
    jogos[
        ctx.channel.id
    ] = objeto_escolhido


    logger.info(
        f"Jogo iniciado | "
        f"Canal: {ctx.channel.id} | "
        f"Objeto: {objeto_escolhido}"
    )


    # --------------------------------------------------------
    # MENSAGEM
    # --------------------------------------------------------

    await ctx.send(
        "🎮 **JOGO DO OBJETO INICIADO!**\n\n"

        f"🎯 Seu objeto é: "
        f"**{NOMES[objeto_escolhido]}**\n\n"

        "📸 Agora envie uma foto junto com:\n"
        "`!adivinhar`"
    )


# ============================================================
# ADIVINHAR
# ============================================================

@bot.command()
async def adivinhar(ctx):

    logger.info(
        f"!adivinhar recebido | "
        f"Usuário: {ctx.author}"
    )


    # --------------------------------------------------------
    # VERIFICAR JOGO
    # --------------------------------------------------------

    if ctx.channel.id not in jogos:

        await ctx.send(
            "❌ Não existe um jogo ativo aqui.\n\n"
            "Comece usando:\n"
            "`!jogodoobjeto`"
        )

        return


    # --------------------------------------------------------
    # VERIFICAR MODELO
    # --------------------------------------------------------

    if model is None:

        await ctx.send(
            "❌ O modelo de inteligência artificial "
            "não foi carregado."
        )

        return


    # --------------------------------------------------------
    # VERIFICAR IMAGEM
    # --------------------------------------------------------

    if not ctx.message.attachments:

        await ctx.send(
            "📸 **Cadê a foto?**\n\n"
            "Envie `!adivinhar` junto com a imagem."
        )

        return


    # --------------------------------------------------------
    # OBJETO ESCOLHIDO
    # --------------------------------------------------------

    objeto_escolhido = jogos[
        ctx.channel.id
    ]


    # --------------------------------------------------------
    # ANEXO
    # --------------------------------------------------------

    attachment = (
        ctx.message.attachments[0]
    )


    # --------------------------------------------------------
    # VERIFICAR EXTENSÃO
    # --------------------------------------------------------

    extensoes = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    )


    if not attachment.filename.lower().endswith(
        extensoes
    ):

        await ctx.send(
            "❌ Envie uma imagem JPG, PNG ou WEBP."
        )

        return


    # --------------------------------------------------------
    # PASTA
    # --------------------------------------------------------

    os.makedirs(
        "downloads",
        exist_ok=True
    )


    # --------------------------------------------------------
    # NOME
    # --------------------------------------------------------

    nome = (
        f"{ctx.message.id}_"
        f"{attachment.filename}"
    )


    caminho = os.path.join(
        "downloads",
        nome
    )


    # --------------------------------------------------------
    # BAIXAR
    # --------------------------------------------------------

    try:

        await attachment.save(
            caminho
        )

    except Exception as erro:

        logger.exception(
            "Erro baixando imagem"
        )

        await ctx.send(
            f"❌ Erro ao baixar imagem:\n"
            f"`{erro}`"
        )

        return


    # --------------------------------------------------------
    # MENSAGEM
    # --------------------------------------------------------

    await ctx.send(
        "🧠 **Analisando a imagem...**\n\n"
        "Estou tentando descobrir o que aparece nela. "
        "Aguarde..."
    )


    # --------------------------------------------------------
    # ANALISAR
    #
    # Usamos uma thread porque o CLIP é pesado
    # e não queremos travar o Discord.
    # --------------------------------------------------------

    try:

        (
            melhor_objeto,
            confianca,
            resultados
        ) = await asyncio.to_thread(
            analisar_imagem,
            caminho,
            objeto_escolhido
        )

    except Exception as erro:

        logger.exception(
            "Erro analisando imagem"
        )

        await ctx.send(
            "❌ **Erro ao analisar a imagem.**\n\n"
            f"`{erro}`"
        )

        try:
            os.remove(caminho)
        except:
            pass

        return


    # ========================================================
    # RESULTADOS
    # ========================================================

    porcentagem = round(
        confianca * 100
    )


    # --------------------------------------------------------
    # PEGAR TOP 5
    # --------------------------------------------------------

    top_resultados = resultados[:5]


    linhas = []

    for objeto, valor in top_resultados:

        pct = round(
            valor * 100
        )

        linhas.append(
            f"{NOMES[objeto]} — **{pct}%**"
        )


    ranking = "\n".join(
        linhas
    )


    # ========================================================
    # ACERTO
    # ========================================================

    if melhor_objeto == objeto_escolhido:

        await ctx.send(
            "🎉🎉 **ACERTOU!** 🎉🎉\n\n"

            f"🎯 Objeto do jogo:\n"
            f"**{NOMES[objeto_escolhido]}**\n\n"

            f"🧠 A IA acha que é:\n"
            f"**{NOMES[melhor_objeto]}**\n\n"

            f"📊 Confiança relativa:\n"
            f"**{porcentagem}%**\n\n"

            "🏆 **VOCÊ GANHOU!** 🏆\n\n"

            "🔎 **Ranking da IA:**\n"
            f"{ranking}"
        )


        logger.info(
            f"ACERTO! "
            f"Esperado={objeto_escolhido} "
            f"Detectado={melhor_objeto}"
        )


        # Finalizar jogo
        del jogos[
            ctx.channel.id
        ]


    # ========================================================
    # ERRO
    # ========================================================

    else:

        await ctx.send(
            "❌ **ERROU!**\n\n"

            f"🎯 O objeto do jogo era:\n"
            f"**{NOMES[objeto_escolhido]}**\n\n"

            f"🧠 A IA acha que a imagem é:\n"
            f"**{NOMES[melhor_objeto]}**\n\n"

            f"📊 Confiança relativa:\n"
            f"**{porcentagem}%**\n\n"

            "🔎 **Ranking da IA:**\n"
            f"{ranking}\n\n"

            "📸 Tente outra imagem!"
        )


        logger.info(
            f"ERRO! "
            f"Esperado={objeto_escolhido} "
            f"Detectado={melhor_objeto}"
        )


    # --------------------------------------------------------
    # APAGAR IMAGEM
    # --------------------------------------------------------

    try:

        os.remove(
            caminho
        )

    except:

        pass


# ============================================================
# RECEBER MENSAGENS
# ============================================================

@bot.event
async def on_message(message):

    # Ignorar mensagens de bots
    if message.author.bot:
        return


    logger.info(
        f"Mensagem recebida | "
        f"{message.author} | "
        f"{message.content} | "
        f"anexos={len(message.attachments)}"
    )


    # Muito importante!
    await bot.process_commands(message)

@bot.command()
async def ajuda(ctx):
    mensagem_ajuda = "\n".join([
        "COMANDOS DISPONIVEIS",
        "",
        "`!ping` - Testa se o bot esta online.",
        "`!hello` - O bot te cumprimenta.",
        "`!jogodoobjeto` - Inicia um jogo do objeto.",
        "`!adivinhar` - Envia uma imagem para adivinhar o objeto.",
        "`!bloons` - Acesso ao modo de dicas do BTD6",
        "`!jogo_código` - Inicia um jogo do código",
        "`ajuda_jogos` - entender sobre o jogos do bot",
    ])
    await ctx.send(mensagem_ajuda)

@bot.command()
async def bloons(ctx):
    await ctx.send(
        "🎮 **MODO DE DICAS DO BTD6**\n\n"
        "Para receber dicas sobre o jogo Bloons TD 6, "
       "biblioteca de comandos:\n"
       "`!bossbtd6` - Dicas sobre os chefes do jogo.\n"
       "`sinergias` - exibe algumas sinergias.\n"
       "`!moabs` - melhores torres/upgrades para moabs.\n"
       "`bloons_n` - melhores torres/upgrades para os bloons normais (não moabs).\n"
       "`algmbtd6` - chamar alguém do servidor para jogar BTD6 com você.\n"
    )

@bot.command()
async def moabs(ctx):

    dicas = [
        "🎮 **MELHORES TORRES/UPGRADES PARA MOABS**",
        
        "1. **Atirador:** pro início dos MOABs eu gosto do 4-0-2, posteriormente sendo colocado no 5-0-2.",
        
        "2. **Pistoleiro:** outro upgrade barato bom é o 0-2-4, posteriormente sendo colocado no 0-2-5.",
        
        "3. **Macaquático:** esse é para rodadas mais avançadas. O 5-0-2 é bem bom.",
        
        "4. **Bumerangue:** esse também é muito bom. O 0-2-5 faz um suporte bem top.",
        
        "5. **Ninja:** considerando o possível farm, tu tem que colocar a Paragon do Ninja."
    ]

    for dica in dicas:
        await ctx.send(dica)
        await asyncio.sleep(2)

@bot.command()
async def bossbtd6(ctx):

    dicas = [
        "🎈 **1 — Bloonarius:** O cospe tachinha 2-0-5 é muito bom pros bloons normais que vêm nos primeiros níveis.",
        "💀 **2 — Lych:** O M.A.D 2-5-0 é bem bom pro Lych.",
        "🌪️ **3 — Vortex:** O Druida 0-2-5 é um excelente counter pro Vortex.",
        "⚡ **4 — Phayze:** O Submarino 5-2-0 é um suporte muito bom pro Phayze.",
        "🪨 **5 — Dreadbloon:** O Super Macaco 2-4-0 é bom para aquelas cerâmicas que ele lança.",
        "🔥 **6 — Blastapopoulos:** O Atirador 5-0-2 é um bom counter pro Blastapopoulos.",
        "🦈 **7 — Diamondback:** A Cola 5-2-0 funciona bem contra ele."
    ]

    for dica in dicas:
        await ctx.send(dica)
        await asyncio.sleep(2)


@bot.command()
async def algmbtd6(ctx):
  await ctx.send(
      f'@everyone Alguém para jogar bloons td 6 com {ctx.author.mention}'
  )

@bot.command()
async def bloons_n(ctx):
    n_d = [
        "1: o atirador 0-2-5 é muito bom pra os bloons normais, principalmente os sem propriedades especiais.(ex:chumbo; ceramica etc...)",
        "2: o pistoleiro 5-2-0 é absurdo pras rodadas antes da 80",
        "3: o cospe cola 5-2-0 é muito bom pra rodadas depois da 80",
        "4: o dardo 0-2-4 junto com o alquimista 4-0-0 é muito bom pra rodadas antes da 60",
    ]

    for dica in n_d:
        await ctx.send(dica)
        await asyncio.sleep(2)


@bot.command()
async def sinergias(ctx):
    sinergias = [
        "🎮 **SINERGIAS**",
        "1. O alquimista 4-0-0 junto com o dardo 0-2-4 é muito bom pra rodadas antes da 60.",
       "2.o druida 0-2-5 junto com alguns druidas 0-1-4 é quebrado",
       "3.o atirador 0-2-5 e o gelo 2-0-5 funciona muito bem juntos",
    ]

    for dica in sinergias:
        await ctx.send(dica)
        await asyncio.sleep(2)


@bot.command()
async def jogo_código(ctx):
    codigo = ''.join(random.choices('0123456789', k=4))

    await ctx.send(
        "🔐 **JOGO DO CÓDIGO**\n\n"
        "Eu gerei um código secreto de **4 dígitos**.\n\n"
        "🟢 Número certo na posição certa\n"
        "🟡 Número certo, mas na posição errada\n"
        "⚫ Número que não existe no código\n\n"
        "🏳️ Digite **desisto** para encerrar e descobrir o código.\n\n"
        "Digite seu palpite:"
    )

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    tentativas = 0

    while True:
        try:
            msg = await bot.wait_for(
                "message",
                check=check,
                timeout=120
            )
        except asyncio.TimeoutError:
            await ctx.send(
                f"⏰ **Tempo esgotado!**\n"
                f"🔐 O código era **{codigo}**."
            )
            return

        palpite = msg.content.lower().strip()

        if palpite == "desisto":
            await ctx.send(
                f"🏳️ **Você desistiu!**\n"
                f"🔐 O código era **{codigo}**."
            )
            return

        if not palpite.isdigit() or len(palpite) != 4:
            await ctx.send(
                "❌ O palpite precisa ter exatamente **4 números**."
            )
            continue

        tentativas += 1

        resultado = ["⚫", "⚫", "⚫", "⚫"]
        codigo_restante = list(codigo)

        # Números na posição certa
        for i in range(4):
            if palpite[i] == codigo[i]:
                resultado[i] = "🟢"
                codigo_restante[i] = None

        # Números certos, mas na posição errada
        for i in range(4):
            if resultado[i] == "🟢":
                continue

            if palpite[i] in codigo_restante:
                resultado[i] = "🟡"
                codigo_restante[codigo_restante.index(palpite[i])] = None

        if resultado == ["🟢", "🟢", "🟢", "🟢"]:
            await ctx.send(
                f"🎉 **ACERTOU!**\n\n"
                f"🔐 Código: **{codigo}**\n"
                f"🎯 Tentativas: **{tentativas}**"
            )
            return

        await ctx.send(
            f"🔎 **Resultado:** {' '.join(resultado)}\n\n"
            f"🟢 = posição certa\n"
            f"🟡 = número certo, posição errada\n"
            f"⚫ = número não existe"
        )

@bot.command()
async def ajuda_jogos(ctx):
    await ctx.send(
        "🎮 **AJUDA SOBRE OS JOGOS DO BOT**\n\n"
        "1️⃣ **Jogo do Objeto:** Use `!jogodoobjeto` para iniciar."
        " Eu escolherei um objeto e você deve enviar uma foto com `!adivinhar` para tentar adivinhar o objeto.\n\n"
        "3️⃣ **Jogo do Código:** Use `!jogo_código` para iniciar. "
        "Tente adivinhar um código secreto de 4 dígitos. O bot informará quantos dígitos estão corretos."
    )
# ============================================================
# INICIAR
# ============================================================

print("Iniciando o bot...")

bot.run(TOKEN)