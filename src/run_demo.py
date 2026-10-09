"""Script de demostración del pipeline de detección de exoplanetas.

Corre el pipeline completo sobre Kepler-10 y genera una figura con 4 paneles:

1. Curva de luz cruda.
2. Curva de luz preprocesada (sin outliers, detrendeada y normalizada).
3. Periodograma BLS con el período detectado marcado.
4. Curva doblada en fase con puntos individuales y bineados.

Uso:
    source venv/bin/activate
    python src/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import download_light_curve
from preprocessing import detrend, normalize_flux, remove_outliers
from transit_detection import find_transits


def main() -> None:
    target = "Kepler-10"
    mission = "Kepler"
    output_figure = Path("docs") / "pipeline_overview.png"
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    print(f"Descargando curva de luz de {target} ({mission})...")
    lc_raw = download_light_curve(target, mission=mission)

    # Para mantener el demo ágil usamos los primeros ~15 días de datos.
    # Kepler-10b tiene P~0.84 d, así que esto aún contiene ~18 tránsitos.
    t_start = lc_raw["time"].min()
    lc_raw = lc_raw[lc_raw["time"] < t_start + 15].copy()

    print("Preprocesando...")
    lc_clean = remove_outliers(lc_raw, sigma=5)
    # Ventana mucho mayor que la duración del tránsito (~0.075 d) para no borrar la señal.
    lc_detrended = detrend(lc_clean, window_length=1001)
    lc_norm = normalize_flux(lc_detrended)

    print("Detectando tránsitos con BLS...")
    bls_result = find_transits(
        lc_norm,
        period_range=(0.5, 1.5),
        output_path=None,  # No guardamos la figura individual de fase aquí.
        n_periods=5000,
        n_durations=40,
    )

    period = bls_result["period"]
    depth = bls_result["depth"]
    duration = bls_result["duration"]
    snr = bls_result["snr"]
    t0 = bls_result["t0"]

    print("\n=== Resultados de detección ===")
    print(f"Período detectado : {period:.5f} d")
    print(f"Profundidad       : {depth:.2e}")
    print(f"Duración          : {duration:.4f} d ({duration * 24:.2f} h)")
    print(f"SNR               : {snr:.1f}")
    print(f"T0                : {t0:.4f} d")

    print("\nGenerando figura...")
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 14), sharex=False, constrained_layout=True
    )

    # --- Panel 1: curva cruda ---
    ax = axes[0]
    ax.scatter(lc_raw["time"], lc_raw["flux"], s=2, alpha=0.4, c="black")
    ax.set_title(f"Curva de luz cruda: {target} ({mission})")
    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Flujo (e⁻/s)")

    # --- Panel 2: preprocesada ---
    ax = axes[1]
    ax.scatter(lc_norm["time"], lc_norm["flux"], s=2, alpha=0.4, c="darkgreen")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Flujo = 1")
    ax.set_title("Curva de luz preprocesada")
    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Flujo normalizado")
    ax.legend()

    # --- Panel 3: periodograma BLS ---
    ax = axes[2]
    ax.plot(bls_result["periods"], bls_result["powers"], "k-", lw=0.8)
    ax.axvline(period, color="red", linestyle="--", linewidth=2, label=f"P = {period:.4f} d")
    ax.set_title("Periodograma Box Least Squares (BLS)")
    ax.set_xlabel("Período (días)")
    ax.set_ylabel("Potencia")
    ax.legend()

    # --- Panel 4: curva doblada en fase ---
    ax = axes[3]
    phase = ((lc_norm["time"].values - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    # Puntos individuales.
    ax.scatter(phase, lc_norm["flux"].values, s=2, alpha=0.3, c="gray", label="Datos")

    # Puntos bineados.
    n_bins = 50
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_flux = np.full_like(bin_centers, np.nan)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.any(mask):
            binned_flux[i] = np.median(lc_norm["flux"].values[mask])
    ax.scatter(bin_centers, binned_flux, s=30, c="red", zorder=3, label="Binned (mediana)")

    ax.set_title("Curva de luz doblada en fase")
    ax.set_xlabel("Fase")
    ax.set_ylabel("Flujo normalizado")
    ax.legend()

    fig.savefig(output_figure, dpi=150)
    print(f"Figura guardada en: {output_figure.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
