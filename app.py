
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from io import BytesIO
from pathlib import Path
from skimage.metrics import structural_similarity
import hashlib


# ==========================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================

st.set_page_config(
    page_title="Pixel Inteligente - SVD",
    page_icon="🦷",
    layout="wide"
)


# ==========================================
# DISEÑO VISUAL
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
        color: #172033;
    }

    h1, h2, h3 {
        color: #143b65;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dce4ef;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# FUNCIONES MATEMÁTICAS
# ==========================================

def cargar_grises(datos):
    imagen = Image.open(BytesIO(datos))
    imagen = ImageOps.exif_transpose(imagen)
    imagen = imagen.convert("L")

    # Limitar dimensiones para mejorar rendimiento
    imagen.thumbnail((800, 800))

    A = np.asarray(imagen, dtype=np.float64)

    return A


def calcular_svd(A):
    U, S, Vt = np.linalg.svd(
        A,
        full_matrices=False
    )

    return U, S, Vt


def reconstruir(U, S, Vt, k):
    Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]

    return np.clip(Ak, 0, 255)


def calcular_metricas(A, Ak, k):
    m, n = A.shape

    original = m * n

    comprimido = k * (m + n + 1)

    razon = original / comprimido

    reduccion = (1 - comprimido / original) * 100

    mse = np.mean((A - Ak) ** 2)

    if mse == 0:
        psnr = np.inf
    else:
        psnr = 10 * np.log10(255 ** 2 / mse)

    ssim = structural_similarity(
        A,
        Ak,
        data_range=255
    )

    return (
        original,
        comprimido,
        razon,
        reduccion,
        mse,
        psnr,
        ssim
    )


# ==========================================
# ENCABEZADO
# ==========================================

st.title("🦷 PIXEL INTELIGENTE")

st.markdown(
    "### Compresión y reconstrucción de radiografías "
    "panorámicas dentales mediante SVD"
)

st.write(
    "Aplicación interactiva desarrollada para el curso "
    "de Álgebra Lineal Computacional."
)

st.divider()


# ==========================================
# PANEL LATERAL
# ==========================================

st.sidebar.title("PIXEL INTELIGENTE")

st.sidebar.header("Configuración")

archivo = st.sidebar.file_uploader(
    "Selecciona una radiografía dental",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.info(
    "Puedes cargar una radiografía o utilizar "
    "la imagen de prueba incluida en el proyecto."
)


# ==========================================
# CARGAR LA RADIOGRAFÍA
# ==========================================

if archivo is not None:

    datos = archivo.getvalue()
    nombre_imagen = archivo.name

else:

    ruta = (
        Path(__file__).parent /
        "radiografia_dental.jpg"
    )

    if not ruta.exists():
        st.error(
            "No se encontró radiografia_dental.jpg. "
            "Carga una imagen desde el panel lateral."
        )
        st.stop()

    datos = ruta.read_bytes()
    nombre_imagen = ruta.name


# ==========================================
# CALCULAR LA DESCOMPOSICIÓN SVD
# ==========================================

identificador = hashlib.sha256(datos).hexdigest()

if st.session_state.get("imagen_id") != identificador:

    A = cargar_grises(datos)

    with st.spinner("Calculando la descomposición SVD..."):
        U, S, Vt = calcular_svd(A)

    st.session_state["imagen_id"] = identificador
    st.session_state["A"] = A
    st.session_state["U"] = U
    st.session_state["S"] = S
    st.session_state["Vt"] = Vt


A = st.session_state["A"]
U = st.session_state["U"]
S = st.session_state["S"]
Vt = st.session_state["Vt"]

m, n = A.shape

max_k = min(400, len(S))


# ==========================================
# CONTROL INTERACTIVO DE K
# ==========================================

st.subheader("Control dinámico de valores singulares")

k = st.slider(
    "Selecciona la cantidad de valores singulares (k)",
    min_value=1,
    max_value=max_k,
    value=min(50, max_k),
    step=1
)

st.write(f"Valor seleccionado: **k = {k}**")


# ==========================================
# RECONSTRUIR RADIOGRAFÍA
# ==========================================

Ak = reconstruir(U, S, Vt, k)

(
    original,
    comprimido,
    razon,
    reduccion,
    mse,
    psnr,
    ssim
) = calcular_metricas(A, Ak, k)


# ==========================================
# VISUALIZACIÓN PRINCIPAL
# ==========================================

col_imagen, col_datos = st.columns([2, 1])

with col_imagen:

    st.subheader("Radiografía reconstruida")

    st.image(
        Ak.astype(np.uint8),
        caption=f"Reconstrucción SVD con k = {k}",
        width="stretch"
    )

    mostrar_original = st.checkbox(
        "Mostrar radiografía original"
    )

    if mostrar_original:

        st.image(
            A.astype(np.uint8),
            caption="Radiografía original",
            width="stretch"
        )


# ==========================================
# PANEL DE MÉTRICAS
# ==========================================

with col_datos:

    st.subheader("Datos de compresión")

    st.write(f"**Imagen:** {nombre_imagen}")

    st.write(f"**Dimensiones:** {m} × {n}")

    st.write(f"**Número de píxeles:** {original:,}")

    st.divider()

    st.metric(
        "Valores singulares (k)",
        k
    )

    st.metric(
        "Coeficientes originales",
        f"{original:,}"
    )

    st.metric(
        "Coeficientes SVD",
        f"{comprimido:,}"
    )

    st.metric(
        "Razón de compresión",
        f"{razon:.2f}:1"
    )

    st.metric(
        "Reducción teórica",
        f"{reduccion:.2f}%"
    )

    if razon < 1:

        st.warning(
            "La representación SVD requiere más "
            "coeficientes que la matriz original."
        )

    st.divider()

    st.subheader("Calidad de reconstrucción")

    st.metric(
        "Error cuadrático medio (MSE)",
        f"{mse:.4f}"
    )

    st.metric(
        "PSNR",
        f"{psnr:.2f} dB"
    )

    st.metric(
        "SSIM",
        f"{ssim:.4f}"
    )


# ==========================================
# GRÁFICA DE VALORES SINGULARES
# ==========================================

st.divider()

st.subheader("Análisis de valores singulares")

mostrar_grafica = st.checkbox(
    "Mostrar gráfica de valores singulares"
)

if mostrar_grafica:

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.semilogy(
        np.arange(1, len(S) + 1),
        S,
        color="blue",
        label="Valores singulares"
    )

    ax.axvline(
        k,
        color="red",
        linestyle="--",
        label=f"k = {k}"
    )

    ax.set_title(
        "Decaimiento de los valores singulares"
    )

    ax.set_xlabel(
        "Índice del valor singular"
    )

    ax.set_ylabel(
        "Magnitud en escala logarítmica"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# FUNDAMENTO MATEMÁTICO
# ==========================================

st.divider()

with st.expander(
    "Ver fundamento matemático de la SVD"
):

    st.subheader(
        "Descomposición en Valores Singulares"
    )

    st.write(
        "La matriz A que representa la radiografía "
        "se descompone mediante:"
    )

    st.latex(
        r"A = U \Sigma V^T"
    )

    st.write(
        "Donde U y V contienen vectores singulares "
        "y Sigma contiene los valores singulares."
    )

    st.subheader(
        "Reconstrucción de rango reducido"
    )

    st.latex(
        r"A_k = U_k \Sigma_k V_k^T"
    )

    st.write(
        "La reconstrucción conserva únicamente "
        "los primeros k valores singulares."
    )

    st.write(
        "De acuerdo con el teorema de "
        "Eckart-Young-Mirsky, la SVD truncada "
        "proporciona una aproximación óptima "
        "de rango menor o igual a k "
        "bajo la norma de Frobenius."
    )

    st.subheader(
        "Error cuadrático medio"
    )

    st.latex(
        r"MSE = \frac{1}{mn}"
        r"\sum_{i=1}^{m}"
        r"\sum_{j=1}^{n}"
        r"(A_{ij}-(A_k)_{ij})^2"
    )

    st.subheader(
        "Razón de compresión"
    )

    st.latex(
        r"RC = \frac{mn}{k(m+n+1)}"
    )


# ==========================================
# PIE DE PÁGINA
# ==========================================

st.divider()

st.caption(
    "PIXEL INTELIGENTE | "
    "Proyecto de Álgebra Lineal Computacional"
)
