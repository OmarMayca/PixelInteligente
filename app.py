
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import hashlib

from PIL import Image, ImageOps
from io import BytesIO
from pathlib import Path
from skimage.metrics import structural_similarity


# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Pixel Inteligente - SVD",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# ESTILOS VISUALES
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
        color: #172033;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #143b65 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 14px;
        border: 1px solid #dce4ef;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #12355b !important;
    }

    div[data-testid="stMetricValue"] > div {
        color: #12355b !important;
    }

    div[data-testid="stMetricValue"] * {
        color: #12355b !important;
    }

    [data-testid="stSidebar"] {
        background-color: #172033;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] button {
        color: #172033 !important;
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

    # Limita dimensiones para mejorar el rendimiento.
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


def calcular_energia(S, k):
    energia_total = np.sum(S ** 2)
    energia_k = np.sum(S[:k] ** 2)

    if energia_total == 0:
        return 100.0

    return 100 * energia_k / energia_total


# ==========================================
# ENCABEZADO
# ==========================================

st.title("🦷 PIXEL INTELIGENTE")

st.markdown(
    "### Compresión y reconstrucción de "
    "radiografías panorámicas mediante SVD"
)

st.write(
    "Proyecto de Álgebra Lineal Computacional. "
    "Aplicación interactiva para evaluar la "
    "compresión y calidad de reconstrucción."
)

st.divider()


# ==========================================
# PANEL LATERAL
# ==========================================

st.sidebar.title("PIXEL INTELIGENTE")

st.sidebar.subheader("Configuración")

archivo = st.sidebar.file_uploader(
    "Selecciona una radiografía",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.info(
    "Puedes cargar una imagen JPG o PNG "
    "o utilizar la radiografía de prueba."
)

st.sidebar.caption(
    "Utiliza imágenes de prueba o radiografías "
    "anonimizadas con autorización."
)


# ==========================================
# CARGA DE IMAGEN
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
            "Carga una radiografía desde el panel lateral."
        )
        st.stop()

    datos = ruta.read_bytes()
    nombre_imagen = ruta.name


# ==========================================
# DESCOMPOSICIÓN SVD
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
# CONTROL DINÁMICO DE K
# ==========================================

st.subheader("Control de valores singulares")

k = st.slider(
    "Selecciona la cantidad de valores singulares (k)",
    min_value=1,
    max_value=max_k,
    value=min(50, max_k),
    step=1
)

st.write(f"Valor seleccionado: **k = {k}**")


# ==========================================
# RECONSTRUCCIÓN Y MÉTRICAS
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

energia = calcular_energia(S, k)


# ==========================================
# COMPARACIÓN DE RADIOGRAFÍAS
# ==========================================

st.subheader("Comparación de radiografías")

col_original, col_reconstruida = st.columns(
    2,
    gap="medium"
)

with col_original:
    st.markdown("**Radiografía original**")

    st.image(
        A.astype(np.uint8),
        caption="Imagen original en escala de grises",
        width="stretch"
    )

with col_reconstruida:
    st.markdown(f"**Reconstrucción SVD - k = {k}**")

    st.image(
        Ak.astype(np.uint8),
        caption=f"Imagen reconstruida con {k} valores singulares",
        width="stretch"
    )


# ==========================================
# INFORMACIÓN DE LA MATRIZ
# ==========================================

st.divider()

st.subheader("Datos de compresión")

st.write(f"**Imagen:** {nombre_imagen}")
st.write(f"**Dimensiones:** {m} × {n}")
st.write(f"**Número de píxeles:** {original:,}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Valores singulares k", k)

with col2:
    st.metric(
        "Coeficientes originales",
        f"{original:,}"
    )

with col3:
    st.metric(
        "Coeficientes SVD",
        f"{comprimido:,}"
    )

with col4:
    st.metric(
        "Razón de compresión",
        f"{razon:.2f}:1"
    )

st.write("**Reducción teórica de coeficientes**")

st.progress(
    min(1.0, max(0.0, reduccion / 100)),
    text=f"{reduccion:.2f}%"
)

if razon < 1:
    st.warning(
        "Con este valor de k, la representación SVD "
        "requiere más coeficientes que la imagen original."
    )


# ==========================================
# MÉTRICAS DE CALIDAD
# ==========================================

st.divider()

st.subheader("Calidad de reconstrucción")

met1, met2, met3, met4 = st.columns(4)

with met1:
    st.metric(
        "Error cuadrático medio",
        f"{mse:.4f}"
    )

with met2:
    st.metric(
        "PSNR",
        f"{psnr:.2f} dB"
    )

with met3:
    st.metric(
        "SSIM",
        f"{ssim:.4f}"
    )

with met4:
    st.metric(
        "Energía conservada",
        f"{energia:.2f}%"
    )

st.caption(
    "MSE, PSNR y SSIM comparan la imagen original "
    "con su reconstrucción. La energía conservada "
    "se calcula a partir de los valores singulares."
)


# ==========================================
# GRÁFICA DE VALORES SINGULARES
# ==========================================

st.divider()

st.subheader("Análisis de valores singulares")

mostrar_grafica = st.checkbox(
    "Mostrar gráfica de valores singulares",
    value=True
)

if mostrar_grafica:

    fig, ax = plt.subplots(figsize=(9, 4))

    indices = np.arange(1, len(S) + 1)

    ax.semilogy(
        indices,
        np.maximum(S, 1e-12),
        color="#2563eb",
        linewidth=2,
        label="Valores singulares"
    )

    ax.axvline(
        k,
        color="#dc2626",
        linestyle="--",
        linewidth=2,
        label=f"k = {k}"
    )

    ax.set_title(
        "Decaimiento de los valores singulares"
    )

    ax.set_xlabel("Índice del valor singular")

    ax.set_ylabel(
        "Magnitud en escala logarítmica"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# FUNDAMENTOS MATEMÁTICOS
# ==========================================

st.divider()

with st.expander("Fundamento matemático del proyecto"):

    st.subheader(
        "Descomposición en Valores Singulares"
    )

    st.write(
        "Una radiografía en escala de grises "
        "puede representarse mediante una matriz A."
    )

    st.latex(r"A = U\Sigma V^T")

    st.write(
        "U contiene los vectores singulares izquierdos, "
        "Sigma contiene los valores singulares y "
        "V transpuesta contiene los vectores "
        "singulares derechos."
    )

    st.subheader("Reconstrucción de rango reducido")

    st.latex(r"A_k = U_k\Sigma_k V_k^T")

    st.write(
        "Se conservan los primeros k valores singulares "
        "para obtener una aproximación de rango reducido."
    )

    st.write(
        "Según el teorema de Eckart-Young-Mirsky, "
        "la SVD truncada proporciona la mejor "
        "aproximación de rango menor o igual a k "
        "bajo la norma de Frobenius y la norma espectral."
    )

    st.subheader("Energía conservada")

    st.latex(
        r"E_k = \frac{\sum_{i=1}^{k}\sigma_i^2}"
        r"{\sum_{i=1}^{r}\sigma_i^2}\times 100\%"
    )

    st.write(
        "Este indicador representa la proporción "
        "de energía matricial conservada al utilizar "
        "los primeros k valores singulares."
    )

    st.subheader("Razón de compresión teórica")

    st.latex(
        r"RC = \frac{mn}{k(m+n+1)}"
    )

    st.write(
        "La razón compara la cantidad de elementos "
        "de la matriz original con los coeficientes "
        "de la representación SVD truncada."
    )

    st.subheader("Error cuadrático medio")

    st.latex(
        r"MSE = \frac{1}{mn}"
        r"\sum_{i=1}^{m}\sum_{j=1}^{n}"
        r"(A_{ij}-(A_k)_{ij})^2"
    )


# ==========================================
# PIE DE PÁGINA
# ==========================================

st.divider()

st.caption(
    "PIXEL INTELIGENTE | "
    "Álgebra Lineal Computacional | "
    "Compresión de radiografías mediante SVD"
)
