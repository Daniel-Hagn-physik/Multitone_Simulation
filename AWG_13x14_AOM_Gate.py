"""
AWG-Ansteuerung fuer das 13x14-Beating-Profil, mit AOM-Gate auf einem dritten Kanal.

Aufbau auf dem Spectrum-Beispiel 2_gen_single_2ch.py (Spectrum Instrumentation GmbH),
erweitert um alles, was fuer die Kameramessung am Ein-Linsen-Aufbau gebraucht wird:

  * 13 x 14 Toene, width 1.56 MHz je Achse  ->  Grundperiode T0 = 100 us
  * Segmentlaenge = ganzzahliges Vielfaches von T0. Nur dann laufen die Toene
    am Schleifenende phasenrichtig weiter; 8 MiS taten das NICHT.
  * Normierung auf die Spitze des fertigen Signals statt auf 1/N. Ohne das
    laeuft der int16-Puffer bei 13 Toenen ueber (Faktor 1.2 bei den
    quadratischen Phasen).
  * Pegel ueber channels.amp(), berechnet aus der gemessenen Kettenverstaerkung
    und der Spitzenleistungsgrenze des AOD.
  * dritter Kanal als AOM-Gate: ein Rechteckpuls von t_gate Laenge an frei
    waehlbarer Position im Segment. Weil alle Kanaele aus demselben Speicher
    und demselben Takt kommen, liegt der Puls sampelgenau zu den Tonphasen -
    das ist der Trigger-Delay t_0 der Simulation.

Warum das Gate ueberhaupt: die Kamera (IDS U3-38J0XLE, IMX415) hat einen
Rolling Shutter mit rund 17.9 us Zeilenzeit. Eine 30-us-Belichtung belichtet
zwar jede Zeile 30 us lang, aber jede Zeile zu einer anderen Zeit - ueber das
Profil verteilt sind das rund zehn Beat-Perioden. Also wird nicht die Kamera
getaktet, sondern das Licht: Belichtung laenger als der Auslesedurchlauf des
ROI (128 Zeilen -> 2.3 ms, also z.B. 5 ms), und in das gemeinsame Zeitfenster
faellt der AOM-Puls.

Der AOM taktet, nicht der AOD: die Toene laufen durch, das Interferenzmuster
steht eingeschwungen, und der Puls schneidet nur ein Stueck heraus. Beim Tasten
der AOD-Toene waeren die ersten rund 5.4 us Einschwingen (Schall-Laufzeit ueber
den Strahl).

Siehe LICENSE des Spectrum-Beispiels fuer dessen Nutzungsbedingungen.
"""

import numpy as np

import spcm
from spcm import units


# ============================================================================
# Parameter
# ============================================================================
SERIAL_NUMBER = 17449          # 4-Kanal-Karte M4i.6622-x8

SAMPLE_RATE = 600e6            # Hz
CH_X, CH_Y, CH_GATE = 2, 3, 0  # Kanalnummern: x-AOD, y-AOD, AOM-Gate

# --- Profil -----------------------------------------------------------------
N_X, N_Y = 13, 14              # Tonzahl je Achse
OFFSET = 100.0e6               # Hz, erster Ton
WIDTH = 1.56e6                 # Hz, Spanne je Achse (beide gleich -> T0 = 100 us)
PHASE_RULE = "quadratic"       # "quadratic" | "schroeder" | "kitayoshi" | "zero"

# --- Pegel ------------------------------------------------------------------
# Spitzenspannung am AWG-Ausgang an 50 Ohm. Herleitung: am AOD sind 4 W
# Momentanspitze erlaubt (Schutzgrenze), die Kette hat 28.37 dB (x) bzw.
# 28.64 dB (y) gemessen am 25.09.2026, also
#     V_pk(AWG) = sqrt(R * C^2 * P_rf / G) = sqrt(R * P_pk / G)
#               = sqrt(50 * 4 / 687) = 0.54 V   bzw. 0.52 V.
# FULL_SCALE ist die Aussteuerung des DAC; der Rest ist Reserve gegen Rundung.
AMP_X = 0.57 * units.V         # = 0.54 V / 0.95
AMP_Y = 0.55 * units.V         # = 0.52 V / 0.95
FULL_SCALE = 0.95              # Anteil des int16-Vollausschlags
CHAIN_GAIN_DB = {"x": 28.37, "y": 28.64}   # gemessen, Filterkette + 5-W-Verstaerker

# --- AOM-Gate ---------------------------------------------------------------
GATE_LENGTH = 30e-6            # s, Belichtungsfenster
GATE_START = 20e-6             # s, Lage im Segment; das ist der Delay t_0.
                               # Mindestens ~10 us, damit der AOD gefuellt ist.
GATE_AMP = 1.0 * units.V       # Pegel fuer den AM-Eingang des AOM-Treibers
GATE_UNIPOLAR = True           # True: 0 .. GATE_AMP (mit Offset), False: +-GATE_AMP
GATE_PER_PERIOD = False        # True: Puls in JEDER Grundperiode (Dauerbetrieb,
                               # z.B. fuer die Photodiode), False: einmal je Segment

# --- Segment ----------------------------------------------------------------
N_PERIODS = 140                # Grundperioden je Segment -> 14 ms bei T0 = 100 us

# --- Trigger ----------------------------------------------------------------
TRIGGER_MODE = "continuous"    # "continuous": Karte laeuft in der Schleife
                               # "camera":     ein Segment je externem Trigger
                               #               (Flash-/Strobe-Ausgang der Kamera
                               #                auf Ext0)


# ============================================================================
# Signal
# ============================================================================
def tone_phases(n_tones, rule=PHASE_RULE):
    """Tonphasen nach der gewaehlten Vorschrift."""
    n = np.arange(n_tones)
    if n_tones <= 1:
        return np.zeros(1)
    if rule == "quadratic":       # phi_n = 2 pi n(n-1)/(N-1)
        return 2.0 * np.pi / (n_tones - 1) * n * (n - 1)
    if rule == "schroeder":       # phi_n = -pi n(n-1)/N
        return -np.pi * n * (n - 1) / n_tones
    if rule == "kitayoshi":
        return np.pi / 2.0 - np.pi * n * (n + 1) / n_tones
    return np.zeros(n_tones)


def tone_frequencies(n_tones, offset=OFFSET, width=WIDTH):
    if n_tones <= 1:
        return np.array([offset + width / 2.0])
    return width * np.arange(n_tones) / (n_tones - 1) + offset


def multitone(t, freqs, phases, full_scale=FULL_SCALE):
    """Mehrtonsignal, auf seine eigene Spitze normiert und als int16.

    Die Normierung auf max|s| statt auf 1/N ist der Punkt: die Aussteuerung
    des DAC haengt dann nicht mehr von Tonzahl und Phasensatz ab, und der
    Pegel wird allein ueber channels.amp() gestellt. Mit 1/N und 13 Toenen
    liegt die Spitze bei 0.46 des Vollausschlags, mit Amplitude 1 je Ton bei
    dem 1.2-fachen - im ersten Fall verschenkt man 6 dB Aufloesung, im
    zweiten laeuft der Puffer ueber."""
    s = np.sin(2 * np.pi * np.asarray(freqs)[None, :] * t[:, None]
               + np.asarray(phases)[None, :]).sum(axis=1)
    peak = np.max(np.abs(s))
    crest = peak / np.sqrt(np.mean(s ** 2))
    s = s / peak * full_scale * (2 ** 15 - 1)
    return s.astype(np.int16), crest


def gate_signal(t, t_start, t_length, period=None, full_scale=FULL_SCALE):
    """Rechteck-Gate fuer den AOM, als int16.

    Das Muster ist +Vollausschlag waehrend des Pulses und -Vollausschlag
    sonst. Mit GATE_UNIPOLAR wird zusaetzlich ein Offset von GATE_AMP gesetzt,
    aus -V..+V wird dann 0..2V am Ausgang - das erwarten die AM-Eingaenge der
    ueblichen AOM-Treiber. Ohne Offset steht das Gate symmetrisch um null,
    was Treiber akzeptieren, die negative Pegel als 'aus' lesen."""
    if period is None:
        on = (t >= t_start) & (t < t_start + t_length)
    else:
        on = (((t - t_start) % period) < t_length) & (t >= t_start)
    hi = full_scale * (2 ** 15 - 1)
    return np.where(on, hi, -hi).astype(np.int16)


def report(freqs_x, freqs_y, crest_x, crest_y, n_samples, t0_period):
    """Was die Einstellungen bedeuten - einmal ausgedruckt, bevor es losgeht."""
    print("Toene x [MHz]:", np.round(freqs_x * 1e-6, 4))
    print("Toene y [MHz]:", np.round(freqs_y * 1e-6, 4))
    print(f"Grundperiode T0 = {t0_period * 1e6:.3f} us, "
          f"Segment = {n_samples} Samples = {n_samples / SAMPLE_RATE * 1e3:.3f} ms "
          f"= {n_samples / SAMPLE_RATE / t0_period:.1f} Perioden")
    for ax, crest, amp in (("x", crest_x, AMP_X), ("y", crest_y, AMP_Y)):
        v_pk = amp.to(units.V).magnitude * FULL_SCALE
        p_mean_awg = (v_pk / crest) ** 2 / 50.0
        g = 10 ** (CHAIN_GAIN_DB[ax] / 10.0)
        p_aod = p_mean_awg * g
        p_pk = crest ** 2 * p_aod
        eta = 0.85 * np.sin(0.5 * np.pi * np.sqrt(min(p_aod / 1.3, 1.0))) ** 2
        print(f"  Achse {ax}: Crest {crest:.2f}, V_pk {v_pk:.3f} V, "
              f"P_AWG {10 * np.log10(p_mean_awg * 1e3):.1f} dBm  ->  "
              f"P_AOD {p_aod:.2f} W (Spitze {p_pk:.2f} W), eta ~ {eta * 100:.0f} %")
        if p_pk > 4.05:
            print(f"  ACHTUNG Achse {ax}: Spitzenleistung {p_pk:.2f} W ueber der "
                  f"Schutzgrenze von 4 W - AMP_{ax.upper()} senken.")


# ============================================================================
# Karte
# ============================================================================
def main():
    t0_period = 1.0 / (WIDTH / np.lcm(N_X - 1, N_Y - 1))     # 100 us
    samples_per_period = SAMPLE_RATE * t0_period             # 60 000
    if abs(samples_per_period - round(samples_per_period)) > 1e-9:
        raise ValueError("Abtastrate und Grundperiode passen nicht zusammen - "
                         "die Segmentlaenge waere nicht ganzzahlig.")
    samples_per_period = int(round(samples_per_period))
    num_samples = samples_per_period * N_PERIODS             # 8 400 000
    if num_samples % 32:
        raise ValueError("Segmentlaenge muss ein Vielfaches von 32 Samples sein.")

    freqs_x, phases_x = tone_frequencies(N_X), tone_phases(N_X)
    freqs_y, phases_y = tone_frequencies(N_Y), tone_phases(N_Y)

    with spcm.Card(serial_number=SERIAL_NUMBER) as card:
        print(card)

        if TRIGGER_MODE == "camera":
            # Ein Segment je Trigger. Der Flash-/Strobe-Ausgang der Kamera
            # markiert das Fenster, in dem alle Zeilen des ROI offen sind.
            card.card_mode(spcm.SPC_REP_STD_SINGLERESTART)
            card.loops(0)
        else:
            card.card_mode(spcm.SPC_REP_STD_CONTINUOUS)
            card.loops(0)

        # Reihenfolge in der Maske: CHANNEL0, CHANNEL2, CHANNEL3.
        ch_mask = (spcm.CHANNEL0 | spcm.CHANNEL2 | spcm.CHANNEL3)
        channels = spcm.Channels(card, card_enable=ch_mask)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        ch_gate, ch_x, ch_y = channels[0], channels[1], channels[2]

        # Pegel je Kanal: die drei Kanaele tragen drei verschiedene Signale.
        try:
            ch_x.amp(AMP_X)
            ch_y.amp(AMP_Y)
            ch_gate.amp(GATE_AMP)
            if GATE_UNIPOLAR:
                ch_gate.offset(GATE_AMP)     # -V..+V wird zu 0..2V
        except Exception as exc:             # aeltere spcm-Version
            print("Pegel je Kanal nicht setzbar (%s) - eine Amplitude fuer alle."
                  % exc.__class__.__name__)
            channels.amp(AMP_X)

        clock = spcm.Clock(card)
        sample_rate = clock.sample_rate(SAMPLE_RATE * units.Hz, return_unit=units.MHz)
        print("Sample rate: {}".format(sample_rate))
        clock.clock_output(False)

        trigger = spcm.Trigger(card)
        if TRIGGER_MODE == "camera":
            trigger.or_mask(spcm.SPC_TMASK_EXT0)
            trigger.ext0_mode(spcm.SPC_TM_POS)
            trigger.ext0_coupling(spcm.COUPLING_DC)
            trigger.ext0_level0(1.5 * units.V)
        else:
            trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

        data_transfer = spcm.DataTransfer(card)
        if data_transfer.bytes_per_sample != 2:
            raise spcm.SpcmException(text="Non 16-bit DA not supported")
        data_transfer.memory_size(num_samples)
        data_transfer.allocate_buffer(num_samples)

        t = np.arange(num_samples) / SAMPLE_RATE
        sig_x, crest_x = multitone(t, freqs_x, phases_x)
        sig_y, crest_y = multitone(t, freqs_y, phases_y)
        gate = gate_signal(t, GATE_START, GATE_LENGTH,
                           period=t0_period if GATE_PER_PERIOD else None)

        report(freqs_x, freqs_y, crest_x, crest_y, num_samples, t0_period)
        print(f"AOM-Gate: {GATE_LENGTH * 1e6:.1f} us ab {GATE_START * 1e6:.1f} us "
              f"({int(round(GATE_LENGTH * SAMPLE_RATE))} Samples ab "
              f"{int(round(GATE_START * SAMPLE_RATE))}), "
              f"{'jede Periode' if GATE_PER_PERIOD else 'einmal je Segment'}")
        print("Delay-Scan: GATE_START in Schritten von 20 us = "
              f"{int(round(20e-6 * SAMPLE_RATE))} Samples verschieben.")

        data_transfer.buffer[ch_x, :] = sig_x
        data_transfer.buffer[ch_y, :] = sig_y
        data_transfer.buffer[ch_gate, :] = gate

        data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA,
                                            spcm.M2CMD_DATA_WAITDMA)

        timeout_time = 30
        card.timeout(timeout_time * units.s)
        print("Starting the card ...")
        try:
            card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
        except spcm.SpcmTimeout:
            print(f"-> Die {timeout_time} s Timeout sind um, die Karte laeuft "
                  f"weiter bzw. wartet auf Trigger.")


if __name__ == "__main__":
    main()
