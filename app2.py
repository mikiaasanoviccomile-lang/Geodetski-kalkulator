import io
import math
import folium
from folium.plugins import MousePosition
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("Geodetski kalkulator: UTM6 vs. Vincenty (WGS84)")


# ==============================================================================
# 1. POMOĆNE FUNKCIJE ZA UGLOVE
# ==============================================================================
def decimalno_u_dms(ugao_deg: float) -> str:
    """Formatira decimalne stepene u čitljiv string (DD° MM' SS.SS'')."""
    znak = "-" if ugao_deg < 0 else ""
    abs_deg = abs(ugao_deg)
    d = int(abs_deg)
    m_float = (abs_deg - d) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return f"{znak}{d}° {m:02d}' {s:05.2f}''"


def decimalno_u_mils(ugao_deg: float, pun_krug_mils: int = 6400) -> float:
    """Pretvara ugao u decimalnim stepenima [0, 360) u hiljadite (mils)."""
    return ((ugao_deg % 360.0) / 360.0) * pun_krug_mils


# ==============================================================================
# 2. UTM6 (KARNEY DIREKTNI I INVERZNI)
# ==============================================================================
def wgs84_to_utm_karney(lat: float, lon: float):
    """Konverzija WGS84 (lat, lon) u UTM6 (easting, northing, zone, hemisphere)."""
    zone = int((lon + 180) / 6) + 1
    lon0 = (zone - 1) * 6 - 180 + 3

    phi = math.radians(lat)
    lam = math.radians(lon)
    lam0 = math.radians(lon0)
    dlam = lam - lam0

    a = 6378137.0
    f = 1 / 298.257223563
    e = math.sqrt(f * (2 - f))
    k0 = 0.9996

    n = f / (2 - f)
    n2, n3, n4, n5, n6 = n**2, n**3, n**4, n**5, n**6
    A = (a / (1 + n)) * (1 + (1 / 4) * n2 + (1 / 64) * n4 + (1 / 256) * n6)

    alpha = [
        0,
        1 / 2 * n - 2 / 3 * n2 + 5 / 16 * n3 + 41 / 180 * n4 - 127 / 288 * n5 + 9361 / 17280 * n6,
        13 / 48 * n2 - 3 / 5 * n3 + 557 / 1440 * n4 + 281 / 630 * n5 - 1983433 / 1935360 * n6,
        61 / 240 * n3 - 103 / 140 * n4 + 15061 / 26880 * n5 + 167603 / 181440 * n6,
        49561 / 161280 * n4 - 179 / 168 * n5 + 6601661 / 7257600 * n6,
        34729 / 80640 * n5 - 3418889 / 1935360 * n6,
        212378941 / 319334400 * n6,
    ]

    tau = math.tan(phi)
    sigma = math.sinh(e * math.atanh(e * tau / math.sqrt(1 + tau**2)))
    tau_prime = tau * math.sqrt(1 + sigma**2) - sigma * math.sqrt(1 + tau**2)

    xi_prime = math.atan2(tau_prime, math.cos(dlam))
    eta_prime = math.asinh(math.sin(dlam) / math.sqrt(tau_prime**2 + math.cos(dlam)**2))

    xi, eta = xi_prime, eta_prime
    for j in range(1, 7):
        two_j = 2 * j
        xi += alpha[j] * math.sin(two_j * xi_prime) * math.cosh(two_j * eta_prime)
        eta += alpha[j] * math.cos(two_j * xi_prime) * math.sinh(two_j * eta_prime)

    easting = 500000.0 + k0 * A * eta
    northing = k0 * A * xi
    hemisphere = "S" if lat < 0 else "N"
    if lat < 0:
        northing += 10000000.0

    return easting, northing, zone, hemisphere


def utm6_to_wgs84_karney(easting: float, northing: float, zone: int, hemisphere: str = "N"):
    """Inverzni Karney (2011) algoritam: pretvara UTM6 (E, N) u WGS84 (lat, lon)."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    e = math.sqrt(e2)
    k0 = 0.9996

    n = f / (2.0 - f)
    n2, n3, n4, n5, n6 = n**2, n**3, n**4, n**5, n**6
    A = (a / (1.0 + n)) * (1.0 + (1.0 / 4.0) * n2 + (1.0 / 64.0) * n4 + (1.0 / 256.0) * n6)

    beta = [
        0.0,
        (1 / 2) * n - (2 / 3) * n2 + (37 / 96) * n3 - (1 / 360) * n4 - (81 / 512) * n5 + (96199 / 604800) * n6,
        (1 / 48) * n2 + (1 / 15) * n3 - (437 / 1440) * n4 + (46 / 105) * n5 - (1118711 / 3870720) * n6,
        (17 / 480) * n3 - (37 / 840) * n4 - (209 / 4480) * n5 + (5569 / 90720) * n6,
        (4397 / 161280) * n4 - (11 / 504) * n5 - (830251 / 7257600) * n6,
        (4583 / 161280) * n5 - (108847 / 3870720) * n6,
        (20648693 / 638668800) * n6,
    ]

    x = easting - 500_000.0
    y = northing if hemisphere.upper() == "N" else northing - 10_000_000.0

    xi = y / (k0 * A)
    eta = x / (k0 * A)

    xi_prime = xi
    eta_prime = eta
    for j in range(1, 7):
        two_j = 2.0 * j
        xi_prime -= beta[j] * math.sin(two_j * xi) * math.cosh(two_j * eta)
        eta_prime -= beta[j] * math.cos(two_j * xi) * math.sinh(two_j * eta)

    lon0_deg = (zone - 1) * 6.0 - 180.0 + 3.0
    dlam_rad = math.atan2(math.sinh(eta_prime), math.cos(xi_prime))
    lon_deg = lon0_deg + math.degrees(dlam_rad)

    tau_prime = math.sin(xi_prime) / math.hypot(math.sinh(eta_prime), math.cos(xi_prime))
    tau = tau_prime

    for _ in range(10):
        sec_phi = math.hypot(1.0, tau)
        sigma = math.sinh(e * math.atanh(e * tau / sec_phi))
        f_tau = tau * math.hypot(1.0, sigma) - sigma * sec_phi
        diff = f_tau - tau_prime
        if abs(diff) < 1e-12:
            break
        dtau_prime_dtau = (math.hypot(1.0, f_tau) * sec_phi) * ((1.0 - e2) / (1.0 + (1.0 - e2) * (tau * tau)))
        tau -= diff / dtau_prime_dtau

    lat_deg = math.degrees(math.atan(tau))
    return lat_deg, lon_deg


def utm6_inverzni(lat1: float, lon1: float, lat2: float, lon2: float):
    """Inverzni problem u UTM6 projekciji."""
    e1, n1, zona, hem = wgs84_to_utm_karney(lat1, lon1)
    e2, n2, _, _ = wgs84_to_utm_karney(lat2, lon2)

    de = e2 - e1
    dn = n2 - n1
    dist = math.hypot(de, dn)
    az1 = (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0
    az2 = (az1 + 180.0) % 360.0
    return dist, az1, az2, (e1, n1), (e2, n2), zona, hem


def utm6_direktni(lat1: float, lon1: float, dist_m: float, heading_deg: float):
    """Direktni problem u UTM6 projekciji: iz (lat1, lon1), rastojanja i headinga daje ciljnu tačku."""
    e1, n1, zona, hem = wgs84_to_utm_karney(lat1, lon1)
    rad = math.radians(heading_deg)
    e2 = e1 + dist_m * math.sin(rad)
    n2 = n1 + dist_m * math.cos(rad)
    lat2, lon2 = utm6_to_wgs84_karney(e2, n2, zona, hem)
    az2 = (heading_deg + 180.0) % 360.0
    return lat2, lon2, az2, (e1, n1), (e2, n2), zona, hem


# ==============================================================================
# 3. VINCENTY ALGORITAM NA WGS84 ELIPSOIDU (DIREKTNI I INVERZNI)
# ==============================================================================
def vincenty_inverzni(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Vincenty (1975) inverzni geodetski zadatak na WGS84 elipsoidu.
    Vraća: (rastojanje_m, pocetni_azimut_deg, povratni_azimut_deg)
    """
    if abs(lat1 - lat2) < 1e-12 and abs(lon1 - lon2) < 1e-12:
        return 0.0, 0.0, 0.0

    a = 6378137.0
    f = 1.0 / 298.257223563
    b = (1.0 - f) * a

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    L = math.radians(lon2 - lon1)

    U1 = math.atan((1.0 - f) * math.tan(phi1))
    U2 = math.atan((1.0 - f) * math.tan(phi2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lam = L
    for _ in range(200):
        sinLam, cosLam = math.sin(lam), math.cos(lam)
        sinSigma = math.sqrt(
            (cosU2 * sinLam) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
        )
        if sinSigma == 0:
            return 0.0, 0.0, 0.0
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = (cosU1 * cosU2 * sinLam) / sinSigma
        cosSqAlpha = 1.0 - sinAlpha**2
        cos2SigmaM = cosSigma - (2.0 * sinU1 * sinU2) / cosSqAlpha if cosSqAlpha != 0 else 0.0

        C = (f / 16.0) * cosSqAlpha * (4.0 + f * (4.0 - 3.0 * cosSqAlpha))
        lamPrev = lam
        lam = L + (1.0 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1.0 + 2.0 * cos2SigmaM**2))
        )
        if abs(lam - lamPrev) < 1e-12:
            break

    uSq = cosSqAlpha * (a**2 - b**2) / (b**2)
    A = 1.0 + (uSq / 16384.0) * (4096.0 + uSq * (-768.0 + uSq * (320.0 - 175.0 * uSq)))
    B = (uSq / 1024.0) * (256.0 + uSq * (-128.0 + uSq * (74.0 - 47.0 * uSq)))
    deltaSigma = (
        B
        * sinSigma
        * (
            cos2SigmaM
            + (B / 4.0)
            * (
                cosSigma * (-1.0 + 2.0 * cos2SigmaM**2)
                - (B / 6.0) * cos2SigmaM * (-3.0 + 4.0 * sinSigma**2) * (-3.0 + 4.0 * cos2SigmaM**2)
            )
        )
    )

    s = b * A * (sigma - deltaSigma)
    alpha1 = math.atan2(cosU2 * math.sin(lam), cosU1 * sinU2 - sinU1 * cosU2 * math.cos(lam))
    alpha2 = math.atan2(cosU1 * math.sin(lam), -sinU1 * cosU2 + cosU1 * sinU2 * math.cos(lam))

    az1_deg = (math.degrees(alpha1) + 360.0) % 360.0
    az2_deg = (math.degrees(alpha2) + 180.0) % 360.0  # Povratni azimut (od cilja ka startu)
    return s, az1_deg, az2_deg


def vincenty_direktni(lat1: float, lon1: float, dist_m: float, azimut1_deg: float):
    """
    Vincenty (1975) direktni geodetski zadatak na WGS84 elipsoidu.
    Vraća: (lat2_deg, lon2_deg, povratni_azimut_deg)
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = (1.0 - f) * a

    phi1 = math.radians(lat1)
    lam1 = math.radians(lon1)
    alpha1 = math.radians(azimut1_deg)

    sinAlpha1 = math.sin(alpha1)
    cosAlpha1 = math.cos(alpha1)

    tanU1 = (1.0 - f) * math.tan(phi1)
    cosU1 = 1.0 / math.sqrt(1.0 + tanU1**2)
    sinU1 = tanU1 * cosU1

    sigma1 = math.atan2(tanU1, cosAlpha1)
    sinAlpha = cosU1 * sinAlpha1
    cosSqAlpha = 1.0 - sinAlpha**2
    uSq = cosSqAlpha * (a**2 - b**2) / (b**2)

    A = 1.0 + (uSq / 16384.0) * (4096.0 + uSq * (-768.0 + uSq * (320.0 - 175.0 * uSq)))
    B = (uSq / 1024.0) * (256.0 + uSq * (-128.0 + uSq * (74.0 - 47.0 * uSq)))

    sigma = dist_m / (b * A)
    for _ in range(200):
        cos2SigmaM = math.cos(2.0 * sigma1 + sigma)
        sinSigma = math.sin(sigma)
        cosSigma = math.cos(sigma)
        deltaSigma = (
            B
            * sinSigma
            * (
                cos2SigmaM
                + (B / 4.0)
                * (
                    cosSigma * (-1.0 + 2.0 * cos2SigmaM**2)
                    - (B / 6.0) * cos2SigmaM * (-3.0 + 4.0 * sinSigma**2) * (-3.0 + 4.0 * cos2SigmaM**2)
                )
            )
        )
        sigmaPrev = sigma
        sigma = (dist_m / (b * A)) + deltaSigma
        if abs(sigma - sigmaPrev) < 1e-12:
            break

    tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
    phi2 = math.atan2(
        sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
        (1.0 - f) * math.sqrt(sinAlpha**2 + tmp**2),
    )
    lam = math.atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1)
    C = (f / 16.0) * cosSqAlpha * (4.0 + f * (4.0 - 3.0 * cosSqAlpha))
    L = lam - (1.0 - C) * f * sinAlpha * (
        sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1.0 + 2.0 * cos2SigmaM**2))
    )
    lam2 = lam1 + L

    alpha2 = math.atan2(sinAlpha, -tmp)
    lat2_deg = math.degrees(phi2)
    lon2_deg = ((math.degrees(lam2) + 540.0) % 360.0) - 180.0
    az2_deg = (math.degrees(alpha2) + 180.0) % 360.0  # Povratni azimut (od cilja ka startu)

    return lat2_deg, lon2_deg, az2_deg


# ==============================================================================
# 4. CSS STILOVI ZA OBOJENE KARTICE (SVETLOPLAVA ZA UTM, SVETLONARANDŽASTA ZA WGS)
# ==============================================================================
st.markdown(
    """
    <style>
    .utm-box {
        background-color: #e3f2fd;
        border-left: 6px solid #1976d2;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #0d2b45;
    }
    .wgs-box {
        background-color: #fff3e0;
        border-left: 6px solid #f57c00;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #4a2500;
    }
    .box-title {
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .box-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    .box-item-label {
        font-size: 12px;
        opacity: 0.8;
    }
    .box-item-val {
        font-size: 18px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 5. INICIJALIZACIJA SESSION STATE-A
# ==============================================================================
if "istorija_utm" not in st.session_state:
    st.session_state.istorija_utm = []
if "istorija_vincenty" not in st.session_state:
    st.session_state.istorija_vincenty = []
if "aktivno_utm" not in st.session_state:
    st.session_state.aktivno_utm = None
if "aktivno_vincenty" not in st.session_state:
    st.session_state.aktivno_vincenty = None


# ==============================================================================
# 6. SIDEBAR: IZBOR ZADATKA (DIREKTNI / INVERZNI) I UNOS PODATAKA
# ==============================================================================
st.sidebar.header("Podešavanja proračuna")

tip_zadatka = st.sidebar.radio(
    "Izaberi geodetski zadatak:",
    options=["Direktni problem (Tačka 1 + Rastojanje + Ugao)", "Inverzni problem (Tačka 1 + Tačka 2)"],
)

mils_standard = st.sidebar.selectbox(
    "Standard za hiljadite (mils):",
    options=[6400, 6000],
    format_func=lambda x: "NATO (6400 mils)" if x == 6400 else "6000 mils (00-00)",
)

with st.sidebar.form("forma_proracun"):
    st.subheader("Tačka 1 (Start - WGS84)")
    naziv1 = st.text_input("Naziv tačke 1:", value="Tačka 1")
    lat1 = st.number_input("Latituda 1 (°):", value=44.786600, format="%.6f", step=0.0001)
    lon1 = st.number_input("Longituda 1 (°):", value=20.448900, format="%.6f", step=0.0001)

    st.markdown("---")

    if tip_zadatka.startswith("Direktni"):
        st.subheader("Parametri pravca (Direktni)")
        naziv2 = st.text_input("Naziv ciljne tačke:", value="Cilj")
        unos_rastojanje = st.number_input("Rastojanje (m):", min_value=0.0, value=5000.0, step=10.0, format="%.2f")
        unos_ugao = st.number_input("Početni ugao / Azimut (°):", min_value=0.0, max_value=360.0, value=45.0, step=0.1, format="%.4f")
    else:
        st.subheader("Tačka 2 (Cilj - WGS84)")
        naziv2 = st.text_input("Naziv tačke 2:", value="Tačka 2")
        lat2_unos = st.number_input("Latituda 2 (°):", value=44.812500, format="%.6f", step=0.0001)
        lon2_unos = st.number_input("Longituda 2 (°):", value=20.461200, format="%.6f", step=0.0001)

    izracunaj = st.form_submit_button("Izračunaj (UTM6 i Vincenty)", use_container_width=True)

if izracunaj:
    rb = len(st.session_state.istorija_utm) + 1

    if tip_zadatka.startswith("Direktni"):
        # 1. UTM6 Direktni
        u_lat2, u_lon2, u_az2, (e1, n1), (e2, n2), zona, _ = utm6_direktni(lat1, lon1, unos_rastojanje, unos_ugao)
        u_mils = decimalno_u_mils(unos_ugao, mils_standard)

        zapis_utm = {
            "Br.": rb,
            "Zadatak": "Direktni",
            "Tačka 1": naziv1,
            "Lat 1": round(lat1, 6),
            "Lon 1": round(lon1, 6),
            "Easting 1 (m)": round(e1, 2),
            "Northing 1 (m)": round(n1, 2),
            "Tačka 2": f"{naziv2} (UTM)",
            "Lat 2": round(u_lat2, 6),
            "Lon 2": round(u_lon2, 6),
            "Easting 2 (m)": round(e2, 2),
            "Northing 2 (m)": round(n2, 2),
            "UTM Zona": zona,
            "Rastojanje (m)": round(unos_rastojanje, 2),
            "Heading (°)": round(unos_ugao, 4),
            "Heading (DMS)": decimalno_u_dms(unos_ugao),
            f"Heading (mils/{mils_standard})": round(u_mils, 2),
            "Povratni ugao (°)": round(u_az2, 4),
        }

        # 2. Vincenty Direktni
        v_lat2, v_lon2, v_az2 = vincenty_direktni(lat1, lon1, unos_rastojanje, unos_ugao)
        v_mils = decimalno_u_mils(unos_ugao, mils_standard)

        zapis_vincenty = {
            "Br.": rb,
            "Zadatak": "Direktni",
            "Tačka 1": naziv1,
            "Lat 1": round(lat1, 6),
            "Lon 1": round(lon1, 6),
            "Tačka 2": f"{naziv2} (WGS84)",
            "Lat 2": round(v_lat2, 6),
            "Lon 2": round(v_lon2, 6),
            "Rastojanje (m)": round(unos_rastojanje, 3),
            "Azimut (°)": round(unos_ugao, 4),
            "Azimut (DMS)": decimalno_u_dms(unos_ugao),
            f"Azimut (mils/{mils_standard})": round(v_mils, 2),
            "Povratni azimut (°)": round(v_az2, 4),
        }

    else:
        # 1. UTM6 Inverzni
        u_dist, u_az1, u_az2, (e1, n1), (e2, n2), zona, _ = utm6_inverzni(lat1, lon1, lat2_unos, lon2_unos)
        u_mils = decimalno_u_mils(u_az1, mils_standard)

        zapis_utm = {
            "Br.": rb,
            "Zadatak": "Inverzni",
            "Tačka 1": naziv1,
            "Lat 1": round(lat1, 6),
            "Lon 1": round(lon1, 6),
            "Easting 1 (m)": round(e1, 2),
            "Northing 1 (m)": round(n1, 2),
            "Tačka 2": naziv2,
            "Lat 2": round(lat2_unos, 6),
            "Lon 2": round(lon2_unos, 6),
            "Easting 2 (m)": round(e2, 2),
            "Northing 2 (m)": round(n2, 2),
            "UTM Zona": zona,
            "Rastojanje (m)": round(u_dist, 2),
            "Heading (°)": round(u_az1, 4),
            "Heading (DMS)": decimalno_u_dms(u_az1),
            f"Heading (mils/{mils_standard})": round(u_mils, 2),
            "Povratni ugao (°)": round(u_az2, 4),
        }

        # 2. Vincenty Inverzni
        v_dist, v_az1, v_az2 = vincenty_inverzni(lat1, lon1, lat2_unos, lon2_unos)
        v_mils = decimalno_u_mils(v_az1, mils_standard)

        zapis_vincenty = {
            "Br.": rb,
            "Zadatak": "Inverzni",
            "Tačka 1": naziv1,
            "Lat 1": round(lat1, 6),
            "Lon 1": round(lon1, 6),
            "Tačka 2": naziv2,
            "Lat 2": round(lat2_unos, 6),
            "Lon 2": round(lon2_unos, 6),
            "Rastojanje (m)": round(v_dist, 3),
            "Azimut (°)": round(v_az1, 4),
            "Azimut (DMS)": decimalno_u_dms(v_az1),
            f"Azimut (mils/{mils_standard})": round(v_mils, 2),
            "Povratni azimut (°)": round(v_az2, 4),
        }

    st.session_state.aktivno_utm = zapis_utm
    st.session_state.aktivno_vincenty = zapis_vincenty
    st.session_state.istorija_utm.append(zapis_utm)
    st.session_state.istorija_vincenty.append(zapis_vincenty)

if st.sidebar.button("Obriši sve podatke", use_container_width=True):
    st.session_state.istorija_utm = []
    st.session_state.istorija_vincenty = []
    st.session_state.aktivno_utm = None
    st.session_state.aktivno_vincenty = None
    st.rerun()


# ==============================================================================
# 7. OBOJENE KARTICE ZA PRIKAZ REZULTATA IZNAD MAPE
# ==============================================================================
akt_u = st.session_state.aktivno_utm
akt_v = st.session_state.aktivno_vincenty

if akt_u and akt_v:
    u_mils_val = decimalno_u_mils(akt_u["Heading (°)"], mils_standard)
    v_mils_val = decimalno_u_mils(akt_v["Azimut (°)"], mils_standard)

    # SVETLOPLAVA KARTICA ZA UTM6
    st.markdown(
        f"""
        <div class="utm-box">
            <div class="box-title">UTM6 Projekcija (Ravanski proračun — Zona {akt_u['UTM Zona']}N)</div>
            <div class="box-grid">
                <div>
                    <div class="box-item-label">Rastojanje (UTM)</div>
                    <div class="box-item-val">{akt_u['Rastojanje (m)']:,.2f} m</div>
                </div>
                <div>
                    <div class="box-item-label">Grid Heading (° / DMS)</div>
                    <div class="box-item-val">{akt_u['Heading (°)']:.4f}° &nbsp;|&nbsp; {akt_u['Heading (DMS)']}</div>
                </div>
                <div>
                    <div class="box-item-label">Heading ({mils_standard} mils)</div>
                    <div class="box-item-val">{u_mils_val:.2f} mils</div>
                </div>
                <div>
                    <div class="box-item-label">Ciljna tačka (WGS84)</div>
                    <div class="box-item-val">{akt_u['Lat 2']:.6f}, {akt_u['Lon 2']:.6f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SVETLONARANDŽASTA KARTICA ZA VINCENTY (WGS84)
    st.markdown(
        f"""
        <div class="wgs-box">
            <div class="box-title">Vincenty — WGS84 Elipsoid (Geodetski proračun)</div>
            <div class="box-grid">
                <div>
                    <div class="box-item-label">Elipsoidno rastojanje</div>
                    <div class="box-item-val">{akt_v['Rastojanje (m)']:,.3f} m</div>
                </div>
                <div>
                    <div class="box-item-label">Geodetski Azimut (° / DMS)</div>
                    <div class="box-item-val">{akt_v['Azimut (°)']:.4f}° &nbsp;|&nbsp; {akt_v['Azimut (DMS)']}</div>
                </div>
                <div>
                    <div class="box-item-label">Azimut ({mils_standard} mils)</div>
                    <div class="box-item-val">{v_mils_val:.2f} mils</div>
                </div>
                <div>
                    <div class="box-item-label">Ciljna tačka (WGS84)</div>
                    <div class="box-item-val">{akt_v['Lat 2']:.6f}, {akt_v['Lon 2']:.6f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Izaberi **Direktni** ili **Inverzni** zadatak sa leve strane, unesi podatke i klikni **Izračunaj**.")


# ==============================================================================
# 8. KREIRANJE I PRIKAZ MAPE
# ==============================================================================
if akt_u:
    centar_lat = (akt_u["Lat 1"] + akt_u["Lat 2"]) / 2.0
    centar_lon = (akt_u["Lon 1"] + akt_u["Lon 2"]) / 2.0
else:
    centar_lat, centar_lon = 44.7866, 20.4489

m = folium.Map(location=[centar_lat, centar_lon], zoom_start=13, tiles="OpenStreetMap")

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    attr="Google Maps",
    name="Google Maps",
).add_to(m)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google Hybrid",
    name="Google Satelit",
).add_to(m)

formatter = "function(num) {return L.Util.formatNum(num, 5) + ' °';};"
MousePosition(
    position="bottomright",
    separator=" | ",
    empty_string="Van mape",
    lng_first=True,
    prefix="WGS84: ",
    lat_formatter=formatter,
    lng_formatter=formatter,
).add_to(m)

if akt_u and akt_v:
    t1_coords = [akt_u["Lat 1"], akt_u["Lon 1"]]
    t2_utm_coords = [akt_u["Lat 2"], akt_u["Lon 2"]]
    t2_wgs_coords = [akt_v["Lat 2"], akt_v["Lon 2"]]

    # Start marker (Zeleni)
    folium.Marker(
        location=t1_coords,
        popup=f"<b>{akt_u['Tačka 1']} (Start)</b><br>Lat: {akt_u['Lat 1']}<br>Lon: {akt_u['Lon 1']}",
        tooltip=f"{akt_u['Tačka 1']} (Start)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    # Cilj UTM6 (Plavi) + linija
    folium.Marker(
        location=t2_utm_coords,
        popup=f"<b>{akt_u['Tačka 2']}</b><br>Lat: {akt_u['Lat 2']}<br>Lon: {akt_u['Lon 2']}",
        tooltip=f"UTM6 Cilj ({akt_u['Rastojanje (m)']} m)",
        icon=folium.Icon(color="blue", icon="flag"),
    ).add_to(m)

    folium.PolyLine(
        locations=[t1_coords, t2_utm_coords],
        color="#1976d2",
        weight=4,
        dash_array="6, 6",
        tooltip=f"UTM6: {akt_u['Rastojanje (m)']} m | Heading: {akt_u['Heading (°)']}°",
    ).add_to(m)

    # Cilj Vincenty WGS84 (Narandžasti) + linija
    folium.Marker(
        location=t2_wgs_coords,
        popup=f"<b>{akt_v['Tačka 2']}</b><br>Lat: {akt_v['Lat 2']}<br>Lon: {akt_v['Lon 2']}",
        tooltip=f"Vincenty Cilj ({akt_v['Rastojanje (m)']} m)",
        icon=folium.Icon(color="orange", icon="star"),
    ).add_to(m)

    folium.PolyLine(
        locations=[t1_coords, t2_wgs_coords],
        color="#f57c00",
        weight=3,
        tooltip=f"Vincenty: {akt_v['Rastojanje (m)']} m | Azimut: {akt_v['Azimut (°)']}°",
    ).add_to(m)

    m.fit_bounds([t1_coords, t2_utm_coords, t2_wgs_coords], padding=(60, 60))

folium.LayerControl().add_to(m)
st_folium(m, width=1400, height=500, returned_objects=[])


# ==============================================================================
# 9. DVE ODVOJENE TABELE ISPOD MAPE + EXCEL IZVOZ
# ==============================================================================
if st.session_state.istorija_utm and st.session_state.istorija_vincenty:
    df_utm = pd.DataFrame(st.session_state.istorija_utm)
    df_wgs = pd.DataFrame(st.session_state.istorija_vincenty)

    st.subheader("1. Tabela: UTM6 Projekcija (Ravanski proračun)")
    st.dataframe(df_utm.iloc[::-1], use_container_width=True, hide_index=True)

    st.subheader("2. Tabela: Vincenty — WGS84 Elipsoid (Geodetski proračun)")
    st.dataframe(df_wgs.iloc[::-1], use_container_width=True, hide_index=True)

    # Excel sa dva odvojena lista (Sheet-a)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_utm.to_excel(writer, index=False, sheet_name="UTM6_Merenja")
        df_wgs.to_excel(writer, index=False, sheet_name="Vincenty_WGS84")

    st.download_button(
        label="Preuzmi obe tabele u Excel formatu (.xlsx)",
        data=buffer.getvalue(),
        file_name="utm6_i_vincenty_merenja.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
