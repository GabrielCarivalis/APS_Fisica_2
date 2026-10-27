import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONSTANTES DO SISTEMA (MODIFIQUE AQUI COMO SE FOSSEM #define EM C)
# ==============================================================================
CAMINHO_VIDEO      = r"C:\Users\Gabriel Carivalis\Trabalho Física\WhatsApp Video 2026-10-26 at 22.21.34.mp4" # Nome do seu ficheiro de vídeo
SAIDA_CSV          = "dados_pendulo_real.csv"                    # Nome da tabela CSV que será gerada
SAIDA_VIDEO_DIAG   = "video_real_rastreado.mp4"                  # Vídeo salvo com a corda virtual e centro

L_METROS           = 0.731                    # Comprimento do fio até o centro da massa (m)
RAIO_BOLINHA_M     = 0.016                    # Raio físico da esfera (m) - p/ momento de inércia
MASSA_KG           = 0.100                    # Massa da esfera (kg)
GRAVIDADE          = 9.79                     # Gravidade local (m/s^2 - Curitiba ~ 9.79)

# Parâmetros de Visão Computacional e Robustez
LIMIAR_CINZA       = None                     # None = Automático (Supressão Cromática HSV) ou fixe (0 a 255)
PESO_SATURACAO_HSV = 1.3                      # Peso para apagar sombras amareladas na parede branca
CIRCULARIDADE_MIN  = 0.70                     # Filtra objetos não esféricos (esfera após ConvexHull ~ 0.85+)
AREA_MIN_PX        = 150                      # Ignora ruídos pequenos na imagem
CORRIGIR_DERIVA_MAO= True                     # True = Remove o movimento lento da câmara se gravado na mão
MOSTRAR_JANELA     = True                     # True = Mostra o vídeo a ser rastreado ao vivo (Q para sair)
SALVAR_VIDEO       = True                     # True = Gera ficheiro MP4 com o rastreamento desenhado
FPS_MANUAL         = None                     # None = Lê o FPS original do vídeo, ou fixe (ex: 30.0 ou 60.0)

# ==============================================================================
# 2. MATEMÁTICA ROBUSTA: REMOÇÃO DE DERIVA DE MÃO, CÍRCULO E SUB-FRAME
# ==============================================================================
def remover_deriva_camera_mao(t, x_px, y_px, dt):
    """
    Se o vídeo foi gravado segurando o telemóvel na mão, o arco inteiro sobe/desce
    lentamente ao longo dos segundos. Esta função rastreia os pontos de equilíbrio
    (passagens pelo fundo do arco) para remover 100% do movimento da mão sem achatar a curvatura!
    """
    dist_min = max(5, int(0.8 / dt))
    prom_x = max(10.0, 0.25 * np.std(x_px))
    idx_dir, _ = find_peaks(x_px, distance=dist_min, prominence=prom_x)
    idx_esq, _ = find_peaks(-x_px, distance=dist_min, prominence=prom_x)

    if len(idx_dir) >= 3 and len(idx_esq) >= 3:
        f_dir = interp1d(t[idx_dir], x_px[idx_dir], kind="linear", fill_value="extrapolate")
        f_esq = interp1d(t[idx_esq], x_px[idx_esq], kind="linear", fill_value="extrapolate")
        deriva_x = 0.5 * (f_dir(t) + f_esq(t))

        cruzamentos = np.where(np.diff(np.sign(x_px - deriva_x)))[0]
        if len(cruzamentos) >= 4:
            f_y = interp1d(t[cruzamentos], y_px[cruzamentos], kind="linear", fill_value="extrapolate")
            deriva_y_bruta = f_y(t)
            janela = min(len(t) - (1 if len(t) % 2 == 0 else 0), max(31, int(2.0 / dt) | 1))
            deriva_y = savgol_filter(deriva_y_bruta, window_length=janela, polyorder=2)
        else:
            deriva_y = np.full_like(y_px, np.mean(y_px))
    else:
        deriva_x = np.full_like(x_px, np.mean(x_px))
        deriva_y = np.full_like(y_px, np.mean(y_px))

    x_estabilizado = x_px - (deriva_x - np.mean(deriva_x))
    y_estabilizado = y_px - (deriva_y - np.mean(deriva_y))
    return x_estabilizado, y_estabilizado, deriva_x, deriva_y

def ajustar_circulo_robusto(x_px, y_px):
    """
    Descobre o pivô invisível (xp, yp) e o comprimento em pixels (R_px):
    1. Chute inicial algébrico exato (Método de Kåsa)
    2. Mínimos Quadrados Não-Lineares com perda 'soft_l1' (imune a outliers)
    3. Filtragem automática de outliers (> 4 desvios medianos) e refinamento final
    """
    A = np.column_stack([2.0 * x_px, 2.0 * y_px, np.ones_like(x_px)])
    b = x_px**2 + y_px**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xp_init, yp_init = c[0], c[1]
    R_init = np.sqrt(max(1.0, c[2] + xp_init**2 + yp_init**2))

    # Garante que o pivô está acima da bolinha (y menor na imagem)
    if yp_init > np.mean(y_px):
        yp_init = np.min(y_px) - R_init

    def residuos(params, x, y):
        xp, yp, R = params
        return np.sqrt((x - xp)**2 + (y - yp)**2) - R

    res1 = least_squares(residuos, [xp_init, yp_init, R_init], args=(x_px, y_px), loss="soft_l1", f_scale=2.0)
    xp, yp, R_px = res1.x

    erros_radiais = np.abs(residuos([xp, yp, R_px], x_px, y_px))
    mad = np.median(erros_radiais) + 1e-6
    inliers = erros_radiais < max(4.0, 4.0 * 1.4826 * mad)

    res2 = least_squares(residuos, [xp, yp, abs(R_px)], args=(x_px[inliers], y_px[inliers]), loss="soft_l1")
    xp, yp, R_px = res2.x
    return xp, yp, abs(R_px), inliers

def detectar_picos_subframe(sinal, tempo, dt, dist_min):
    """
    Usa interpolação parabólica nos 3 pontos ao redor de cada pico para achar
    o instante exato entre dois frames da câmara (precisão sub-milissegundo).
    """
    prom_min = max(0.01, 0.15 * np.std(sinal))
    idx, _ = find_peaks(sinal, distance=dist_min, prominence=prom_min)
    t_sub, val_sub = [], []
    for i in idx:
        if 0 < i < len(sinal) - 1:
            y0, y1, y2 = sinal[i-1], sinal[i], sinal[i+1]
            denom = (y0 - 2.0 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            delta = np.clip(delta, -1.0, 1.0)
            t_sub.append(tempo[i] + delta * dt)
            val_sub.append(y1 - 0.25 * (y0 - y2) * delta)
        else:
            t_sub.append(tempo[i])
            val_sub.append(sinal[i])
    return np.array(t_sub), np.array(val_sub)

def envelope_amortecimento(t, theta0, gamma):
    return theta0 * np.exp(-gamma * t)

def periodo_grandes_angulos(theta_rad, T0):
    return T0 * (1.0 + (1.0/16.0)*(theta_rad**2) + (11.0/3072.0)*(theta_rad**4) + (173.0/737280.0)*(theta_rad**6))

# ==============================================================================
# 3. ETAPA 1: VISÃO COMPUTACIONAL COM SUPRESSÃO CROMÁTICA DE SOMBRA (HSV)
# ==============================================================================
def extrair_posicoes_video(caminho):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {caminho}")

    fps = FPS_MANUAL if FPS_MANUAL is not None else cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    dt = 1.0 / fps

    frames_idx, tempos, x_pixels, y_pixels, raios_px = [], [], [], [], []
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if LIMIAR_CINZA is None:
            # Supressão Cromática de Sombra: a sombra projetada na parede tem saturação S alta (quente/amarelada),
            # enquanto a bolinha metálica/cinzenta tem saturação S baixa (neutra).
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            S = hsv[:, :, 1].astype(np.float32)
            gray_sem_sombra = np.clip(gray.astype(np.float32) + PESO_SATURACAO_HSV * S, 0, 255).astype(np.uint8)
            blur = cv2.GaussianBlur(gray_sem_sombra, (5, 5), 0)

            tom_fundo = float(np.median(blur))
            tom_min   = float(np.percentile(blur, 0.2))
            limiar_auto = int(tom_min + 0.45 * (tom_fundo - tom_min))
            _, mask_raw = cv2.threshold(blur, limiar_auto, 255, cv2.THRESH_BINARY_INV)
        else:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask_raw = cv2.threshold(blur, LIMIAR_CINZA, 255, cv2.THRESH_BINARY_INV)

        # Fecha o reflexo metálico brilhante no meio da esfera e remove penumbra externa
        mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, k_close)
        mask_clean  = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, k_open)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        melhor_hull  = None
        melhor_score = 0.0

        for cnt in contours:
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            if area < AREA_MIN_PX:
                continue
            perimetro = cv2.arcLength(hull, True)
            if perimetro <= 0:
                continue
            circularidade = 4.0 * np.pi * area / (perimetro ** 2)
            if circularidade >= CIRCULARIDADE_MIN:
                score = area * (circularidade ** 2)
                if score > melhor_score:
                    melhor_score = score
                    melhor_hull = hull

        if melhor_hull is not None:
            M = cv2.moments(melhor_hull)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                r_bolinha = np.sqrt(M["m00"] / np.pi)
                frames_idx.append(frame_count)
                tempos.append(frame_count * dt)
                x_pixels.append(cx)
                y_pixels.append(cy)
                raios_px.append(r_bolinha)

        frame_count += 1

    cap.release()
    return (np.array(frames_idx), np.array(tempos), np.array(x_pixels),
            np.array(y_pixels), np.array(raios_px), fps, dt, frame_count)

# ==============================================================================
# 4. RENDERIZAÇÃO DO VÍDEO DE DIAGNÓSTICO COM CORDA VIRTUAL E CENTRO DE MASSA
# ==============================================================================
def gerar_video_diagnostico(caminho_entrada, caminho_saida, frames_full, x_orig, y_orig, xp_din, yp_din, r_med_px, fps):
    if not SALVAR_VIDEO and not MOSTRAR_JANELA:
        return
    cap = cv2.VideoCapture(caminho_entrada)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if SALVAR_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(caminho_saida, fourcc, fps, (w, h))

    mapa_pos = {
        int(f): (float(x), float(y), float(px_p), float(py_p))
        for f, x, y, px_p, py_p in zip(frames_full, x_orig, y_orig, xp_din, yp_din)
    }
    rastro = []
    f_idx = 0
    raio_vis = int(round(r_med_px))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx in mapa_pos:
            cx, cy, p_x, p_y = mapa_pos[f_idx]
            rastro.append((int(cx), int(cy)))
            if len(rastro) > 45:
                rastro.pop(0)
            # Desenha o rastro recente da trajetória
            for k in range(1, len(rastro)):
                cv2.line(frame, rastro[k-1], rastro[k], (255, 180, 0), 2, cv2.LINE_AA)
            # Desenha a corda virtual reconstruída até ao pivô (mesmo fora do ecrã!)
            cv2.line(frame, (int(p_x), int(p_y)), (int(cx), int(cy)), (255, 80, 0), 2, cv2.LINE_AA)
            # Desenha o contorno da esfera e o centro de massa cravado
            cv2.circle(frame, (int(cx), int(cy)), raio_vis, (0, 230, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        if MOSTRAR_JANELA:
            escala_vis = min(1.0, 900.0 / max(w, h))
            frame_vis = cv2.resize(frame, (int(w * escala_vis), int(h * escala_vis))) if escala_vis < 1.0 else frame
            cv2.imshow("Rastreamento do Pendulo (Pressione Q para sair)", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        f_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

# ==============================================================================
# 5. ETAPA 2: CALIBRAÇÃO FÍSICA, TABELA COMPLETA E GRÁFICOS
# ==============================================================================
def analisar_pendulo():
    frames_raw, t_raw, x_raw, y_raw, raios_px, fps, dt, total_frames = extrair_posicoes_video(CAMINHO_VIDEO)
    if len(t_raw) < 15:
        raise ValueError("Poucos frames rastreados! Verifique o caminho do vídeo ou ajuste LIMIAR_CINZA.")

    # 1. Interpola grade uniforme (caso algum frame tenha falhado)
    frames_full = np.arange(frames_raw[0], frames_raw[-1] + 1)
    t_full = (frames_full - frames_full[0]) * dt
    x_orig = np.interp(frames_full, frames_raw, x_raw)
    y_orig = np.interp(frames_full, frames_raw, y_raw)
    r_med_px = float(np.median(raios_px))

    # 2. Estabiliza deriva lenta caso a câmara tenha sido segurada na mão
    if CORRIGIR_DERIVA_MAO:
        x_est, y_est, deriva_x, deriva_y = remover_deriva_camera_mao(t_full, x_orig, y_orig, dt)
    else:
        x_est, y_est = x_orig.copy(), y_orig.copy()
        deriva_x = np.full_like(x_orig, np.mean(x_orig))
        deriva_y = np.full_like(y_orig, np.mean(y_orig))

    # 3. Encontra o pivô invisível (xp, yp) e o raio do arco em pixels (R_px)
    xp_px, yp_px, R_px, inliers = ajustar_circulo_robusto(x_est, y_est)

    # Pivô dinâmico no referencial original do vídeo (acompanha o movimento da mão para desenhar a corda certa)
    xp_din = xp_px + (deriva_x - np.mean(deriva_x))
    yp_din = yp_px + (deriva_y - np.mean(deriva_y))

    # 4. Comprimento efetivo e escala física (metros/pixel)
    L_efetivo = L_METROS + (2.0 * (RAIO_BOLINHA_M ** 2)) / (5.0 * L_METROS) if RAIO_BOLINHA_M > 0 else L_METROS
    escala_m_por_px = L_METROS / R_px

    dx_m = (x_est - xp_px) * escala_m_por_px
    dy_m = -(y_est - yp_px) * escala_m_por_px
    theta_bruto = np.arctan2(dx_m, -dy_m)

    # Corrige inclinação média da câmara em relação à gravidade
    theta_eq = 0.5 * (np.percentile(theta_bruto, 98) + np.percentile(theta_bruto, 2))
    theta_rad = theta_bruto - theta_eq
    theta_deg = np.degrees(theta_rad)

    # Coordenadas físicas exatas em metros (com origem (0,0) no pivô)
    x_m = L_METROS * np.sin(theta_rad)
    y_m = -L_METROS * np.cos(theta_rad)

    # 5. Velocidades e Energias via Filtro Polinomial de Savitzky-Golay
    janela = min(15, len(t_full) - (1 if len(t_full) % 2 == 0 else 0))
    if janela >= 5:
        vx = savgol_filter(x_m, window_length=janela, polyorder=3, deriv=1, delta=dt)
        vy = savgol_filter(y_m, window_length=janela, polyorder=3, deriv=1, delta=dt)
        omega_rad = savgol_filter(theta_rad, window_length=janela, polyorder=3, deriv=1, delta=dt)
        theta_suave = savgol_filter(theta_rad, window_length=janela, polyorder=3)
    else:
        vx = np.gradient(x_m, t_full)
        vy = np.gradient(y_m, t_full)
        omega_rad = np.gradient(theta_rad, t_full)
        theta_suave = theta_rad

    v_total = np.sqrt(vx**2 + vy**2)
    E_cinetica  = 0.5 * MASSA_KG * (v_total**2)
    E_potencial = MASSA_KG * GRAVIDADE * L_METROS * (1.0 - np.cos(theta_rad))
    E_mecanica  = E_cinetica + E_potencial

    # 6. Salva a Tabela Completa (CSV)
    tabela = pd.DataFrame({
        "frame": frames_full,
        "t_s": np.round(t_full, 5),
        "x_px": np.round(x_orig, 2),
        "y_px": np.round(y_orig, 2),
        "x_estab_px": np.round(x_est, 2),
        "y_estab_px": np.round(y_est, 2),
        "x_m": np.round(x_m, 5),
        "y_m": np.round(y_m, 5),
        "theta_rad": np.round(theta_rad, 5),
        "theta_graus": np.round(theta_deg, 3),
        "vx_m_s": np.round(vx, 5),
        "vy_m_s": np.round(vy, 5),
        "omega_rad_s": np.round(omega_rad, 5),
        "E_cinetica_J": np.round(E_cinetica, 6),
        "E_potencial_J": np.round(E_potencial, 6),
        "E_mecanica_J": np.round(E_mecanica, 6)
    })
    tabela.to_csv(SAIDA_CSV, index=False)

    # 7. Detecção de Picos Sub-Frame, Período, Amortecimento e Fator Q
    T0_teorico = 2.0 * np.pi * np.sqrt(L_efetivo / GRAVIDADE)
    dist_min = max(5, int(0.6 * T0_teorico / dt))

    t_cristas, th_cristas = detectar_picos_subframe(theta_suave, t_full, dt, dist_min)
    t_vales, th_vales     = detectar_picos_subframe(-theta_suave, t_full, dt, dist_min)

    t_extremos  = np.concatenate([t_cristas, t_vales])
    th_extremos = np.concatenate([th_cristas, th_vales])
    ordem = np.argsort(t_extremos)
    t_extremos, th_extremos = t_extremos[ordem], th_extremos[ordem]

    periodos_ciclos = np.diff(t_cristas)
    periodo_medio   = float(np.mean(periodos_ciclos))
    periodo_std     = float(np.std(periodos_ciclos))
    omega_d         = 2.0 * np.pi / periodo_medio

    # Comprimento físico implícito pelo período real medido no vídeo
    theta_rms = float(np.mean(th_extremos))
    T0_medido = periodo_medio / (1.0 + (1.0/16.0)*(theta_rms**2))
    L_implicito_video = GRAVIDADE * ((T0_medido / (2.0 * np.pi)) ** 2)

    # Ajuste exponencial robusto (garante gamma >= 0)
    try:
        popt, _ = curve_fit(
            envelope_amortecimento, t_extremos, th_extremos,
            p0=[th_extremos[0], 0.005],
            bounds=([0.0, 1e-6], [np.pi, 2.0])
        )
        theta0_fit, gamma_exp = popt
    except Exception:
        theta0_fit, gamma_exp = float(th_extremos[0]), 0.001

    b_viscoso      = 2.0 * MASSA_KG * gamma_exp
    tau_relax      = 1.0 / gamma_exp
    decremento_log = gamma_exp * periodo_medio
    fator_Q        = omega_d / (2.0 * gamma_exp)

    print("\n" + "="*68)
    print("        RELATÓRIO FÍSICO DO PÊNDULO REAL (VERSÃO FINAL)")
    print("="*68)
    print(f"Frames detetados com sucesso : {len(frames_raw)} de {total_frames} ({len(frames_raw)/total_frames*100:.1f}%)")
    print(f"Raio médio da esfera no vídeo: {r_med_px:.2f} px (Diâmetro = {2*r_med_px:.1f} px)")
    print(f"Pivô invisível reconstruído  : ({xp_px:.1f}, {yp_px:.1f}) px | Raio Arco = {R_px:.1f} px")
    print(f"Comprimento configurado (L)  : {L_METROS:.4f} m (L implícito pelo período = {L_implicito_video:.4f} m)")
    print("-" * 68)
    print(f"Número de ciclos completos   : {len(periodos_ciclos)} oscilações")
    print(f"Ângulo Inicial (θ0)          : {np.degrees(theta0_fit):.2f} graus ({theta0_fit:.3f} rad)")
    print(f"Período Linear Teórico (T0)  : {T0_teorico:.4f} s")
    print(f"Período 1º Ciclo             : {periodos_ciclos[0]:.4f} s")
    print(f"Período Último Ciclo         : {periodos_ciclos[-1]:.4f} s")
    print(f"Período Médio Experimental   : {periodo_medio:.4f} +/- {periodo_std:.4f} s")
    print(f"Frequência Angular (ω_d)     : {omega_d:.4f} rad/s")
    print("-" * 68)
    print(f"Coef. de Amortecimento (γ)   : {gamma_exp:.5f} s^-1")
    print(f"Constante Viscosa (b = 2mγ)  : {b_viscoso:.6f} kg/s")
    print(f"Tempo de Relaxação (τ = 1/γ) : {tau_relax:.2f} s")
    print(f"Decremento Logarítmico (δ)   : {decremento_log:.5f}")
    print(f"FATOR DE QUALIDADE (Q)       : {fator_Q:.1f}")
    print("="*68)

    gerar_video_diagnostico(CAMINHO_VIDEO, SAIDA_VIDEO_DIAG, frames_full, x_orig, y_orig, xp_din, yp_din, r_med_px, fps)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(t_full, theta_deg, label="Ângulo Experimental θ(t)", color="royalblue")
    axs[0].plot(t_cristas, np.degrees(th_cristas), "ro", markersize=4, label="Cristas/Vales Detetados")
    axs[0].plot(t_vales, -np.degrees(th_vales), "ro", markersize=4)
    axs[0].plot(t_full, np.degrees(envelope_amortecimento(t_full, theta0_fit, gamma_exp)), "r--",
                label=f"Envelope: γ = {gamma_exp:.5f} s^-1 | Q = {fator_Q:.1f}")
    axs[0].plot(t_full, -np.degrees(envelope_amortecimento(t_full, theta0_fit, gamma_exp)), "r--")
    axs[0].set_ylabel("Ângulo θ (graus)")
    axs[0].set_title("Oscilação do Pêndulo Real e Envelope de Amortecimento")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    ciclos_num = np.arange(1, len(periodos_ciclos) + 1)
    axs[1].plot(ciclos_num, periodos_ciclos, "o-", color="darkorange", label="Período Medido por Ciclo")
    axs[1].axhline(periodo_medio, color="black", linestyle="--", label=f"Período Médio = {periodo_medio:.4f} s")
    axs[1].set_xlabel("Número do Ciclo")
    axs[1].set_ylabel("Período T (s)")
    axs[1].set_title("Estabilidade do Período Ciclo a Ciclo")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_full, E_cinetica, label="Energia Cinética (K)", alpha=0.6, color="green")
    axs[2].plot(t_full, E_potencial, label="Energia Potencial (U)", alpha=0.6, color="purple")
    axs[2].plot(t_full, E_mecanica, label="Energia Mecânica Total (E = K + U)", color="black", linewidth=1.5)
    axs[2].set_xlabel("Tempo t (s)")
    axs[2].set_ylabel("Energia (Joules)")
    axs[2].set_title("Energias Cinética, Potencial e Mecânica Total (Estabilizada)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("diagnostico_pendulo_real.png", dpi=150)
    if MOSTRAR_JANELA:
        plt.show()

if __name__ == "__main__":
    analisar_pendulo()