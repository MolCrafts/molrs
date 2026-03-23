//! Element data and basic lookup utilities.
use core::str::FromStr;

/// Chemical element with complete periodic table (elements 1-118)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Element {
    H = 1,
    He = 2,
    Li = 3,
    Be = 4,
    B = 5,
    C = 6,
    N = 7,
    O = 8,
    F = 9,
    Ne = 10,
    Na = 11,
    Mg = 12,
    Al = 13,
    Si = 14,
    P = 15,
    S = 16,
    Cl = 17,
    Ar = 18,
    K = 19,
    Ca = 20,
    Sc = 21,
    Ti = 22,
    V = 23,
    Cr = 24,
    Mn = 25,
    Fe = 26,
    Co = 27,
    Ni = 28,
    Cu = 29,
    Zn = 30,
    Ga = 31,
    Ge = 32,
    As = 33,
    Se = 34,
    Br = 35,
    Kr = 36,
    Rb = 37,
    Sr = 38,
    Y = 39,
    Zr = 40,
    Nb = 41,
    Mo = 42,
    Tc = 43,
    Ru = 44,
    Rh = 45,
    Pd = 46,
    Ag = 47,
    Cd = 48,
    In = 49,
    Sn = 50,
    Sb = 51,
    Te = 52,
    I = 53,
    Xe = 54,
    Cs = 55,
    Ba = 56,
    La = 57,
    Ce = 58,
    Pr = 59,
    Nd = 60,
    Pm = 61,
    Sm = 62,
    Eu = 63,
    Gd = 64,
    Tb = 65,
    Dy = 66,
    Ho = 67,
    Er = 68,
    Tm = 69,
    Yb = 70,
    Lu = 71,
    Hf = 72,
    Ta = 73,
    W = 74,
    Re = 75,
    Os = 76,
    Ir = 77,
    Pt = 78,
    Au = 79,
    Hg = 80,
    Tl = 81,
    Pb = 82,
    Bi = 83,
    Po = 84,
    At = 85,
    Rn = 86,
    Fr = 87,
    Ra = 88,
    Ac = 89,
    Th = 90,
    Pa = 91,
    U = 92,
    Np = 93,
    Pu = 94,
    Am = 95,
    Cm = 96,
    Bk = 97,
    Cf = 98,
    Es = 99,
    Fm = 100,
    Md = 101,
    No = 102,
    Lr = 103,
    Rf = 104,
    Db = 105,
    Sg = 106,
    Bh = 107,
    Hs = 108,
    Mt = 109,
    Ds = 110,
    Rg = 111,
    Cn = 112,
    Nh = 113,
    Fl = 114,
    Mc = 115,
    Lv = 116,
    Ts = 117,
    Og = 118,
}

/// Element properties data structure
struct ElementData {
    symbol: &'static str,
    name: &'static str,
    mass: f32,
    covalent_radius: f32, // in Angstroms
    vdw_radius: f32,      // in Angstroms
}

// Element data table (indexed by atomic number - 1)
const ELEMENT_DATA: [ElementData; 118] = [
    ElementData {
        symbol: "H",
        name: "Hydrogen",
        mass: 1.008,
        covalent_radius: 0.31,
        vdw_radius: 1.20,
    },
    ElementData {
        symbol: "He",
        name: "Helium",
        mass: 4.003,
        covalent_radius: 0.28,
        vdw_radius: 1.40,
    },
    ElementData {
        symbol: "Li",
        name: "Lithium",
        mass: 6.941,
        covalent_radius: 1.28,
        vdw_radius: 1.82,
    },
    ElementData {
        symbol: "Be",
        name: "Beryllium",
        mass: 9.012,
        covalent_radius: 0.96,
        vdw_radius: 1.53,
    },
    ElementData {
        symbol: "B",
        name: "Boron",
        mass: 10.81,
        covalent_radius: 0.84,
        vdw_radius: 1.92,
    },
    ElementData {
        symbol: "C",
        name: "Carbon",
        mass: 12.01,
        covalent_radius: 0.76,
        vdw_radius: 1.70,
    },
    ElementData {
        symbol: "N",
        name: "Nitrogen",
        mass: 14.01,
        covalent_radius: 0.71,
        vdw_radius: 1.55,
    },
    ElementData {
        symbol: "O",
        name: "Oxygen",
        mass: 16.00,
        covalent_radius: 0.66,
        vdw_radius: 1.52,
    },
    ElementData {
        symbol: "F",
        name: "Fluorine",
        mass: 19.00,
        covalent_radius: 0.57,
        vdw_radius: 1.47,
    },
    ElementData {
        symbol: "Ne",
        name: "Neon",
        mass: 20.18,
        covalent_radius: 0.58,
        vdw_radius: 1.54,
    },
    ElementData {
        symbol: "Na",
        name: "Sodium",
        mass: 22.99,
        covalent_radius: 1.66,
        vdw_radius: 2.27,
    },
    ElementData {
        symbol: "Mg",
        name: "Magnesium",
        mass: 24.31,
        covalent_radius: 1.41,
        vdw_radius: 1.73,
    },
    ElementData {
        symbol: "Al",
        name: "Aluminum",
        mass: 26.98,
        covalent_radius: 1.21,
        vdw_radius: 1.84,
    },
    ElementData {
        symbol: "Si",
        name: "Silicon",
        mass: 28.09,
        covalent_radius: 1.11,
        vdw_radius: 2.10,
    },
    ElementData {
        symbol: "P",
        name: "Phosphorus",
        mass: 30.97,
        covalent_radius: 1.07,
        vdw_radius: 1.80,
    },
    ElementData {
        symbol: "S",
        name: "Sulfur",
        mass: 32.07,
        covalent_radius: 1.05,
        vdw_radius: 1.80,
    },
    ElementData {
        symbol: "Cl",
        name: "Chlorine",
        mass: 35.45,
        covalent_radius: 1.02,
        vdw_radius: 1.75,
    },
    ElementData {
        symbol: "Ar",
        name: "Argon",
        mass: 39.95,
        covalent_radius: 1.06,
        vdw_radius: 1.88,
    },
    ElementData {
        symbol: "K",
        name: "Potassium",
        mass: 39.10,
        covalent_radius: 2.03,
        vdw_radius: 2.75,
    },
    ElementData {
        symbol: "Ca",
        name: "Calcium",
        mass: 40.08,
        covalent_radius: 1.76,
        vdw_radius: 2.31,
    },
    ElementData {
        symbol: "Sc",
        name: "Scandium",
        mass: 44.96,
        covalent_radius: 1.70,
        vdw_radius: 2.11,
    },
    ElementData {
        symbol: "Ti",
        name: "Titanium",
        mass: 47.87,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "V",
        name: "Vanadium",
        mass: 50.94,
        covalent_radius: 1.53,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Cr",
        name: "Chromium",
        mass: 52.00,
        covalent_radius: 1.39,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Mn",
        name: "Manganese",
        mass: 54.94,
        covalent_radius: 1.39,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Fe",
        name: "Iron",
        mass: 55.85,
        covalent_radius: 1.32,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Co",
        name: "Cobalt",
        mass: 58.93,
        covalent_radius: 1.26,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ni",
        name: "Nickel",
        mass: 58.69,
        covalent_radius: 1.24,
        vdw_radius: 1.63,
    },
    ElementData {
        symbol: "Cu",
        name: "Copper",
        mass: 63.55,
        covalent_radius: 1.32,
        vdw_radius: 1.40,
    },
    ElementData {
        symbol: "Zn",
        name: "Zinc",
        mass: 65.39,
        covalent_radius: 1.22,
        vdw_radius: 1.39,
    },
    ElementData {
        symbol: "Ga",
        name: "Gallium",
        mass: 69.72,
        covalent_radius: 1.22,
        vdw_radius: 1.87,
    },
    ElementData {
        symbol: "Ge",
        name: "Germanium",
        mass: 72.61,
        covalent_radius: 1.20,
        vdw_radius: 2.11,
    },
    ElementData {
        symbol: "As",
        name: "Arsenic",
        mass: 74.92,
        covalent_radius: 1.19,
        vdw_radius: 1.85,
    },
    ElementData {
        symbol: "Se",
        name: "Selenium",
        mass: 78.96,
        covalent_radius: 1.20,
        vdw_radius: 1.90,
    },
    ElementData {
        symbol: "Br",
        name: "Bromine",
        mass: 79.90,
        covalent_radius: 1.20,
        vdw_radius: 1.85,
    },
    ElementData {
        symbol: "Kr",
        name: "Krypton",
        mass: 83.80,
        covalent_radius: 1.16,
        vdw_radius: 2.02,
    },
    ElementData {
        symbol: "Rb",
        name: "Rubidium",
        mass: 85.47,
        covalent_radius: 2.20,
        vdw_radius: 3.03,
    },
    ElementData {
        symbol: "Sr",
        name: "Strontium",
        mass: 87.62,
        covalent_radius: 1.95,
        vdw_radius: 2.49,
    },
    ElementData {
        symbol: "Y",
        name: "Yttrium",
        mass: 88.91,
        covalent_radius: 1.90,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Zr",
        name: "Zirconium",
        mass: 91.22,
        covalent_radius: 1.75,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Nb",
        name: "Niobium",
        mass: 92.91,
        covalent_radius: 1.64,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Mo",
        name: "Molybdenum",
        mass: 95.94,
        covalent_radius: 1.54,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Tc",
        name: "Technetium",
        mass: 98.00,
        covalent_radius: 1.47,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ru",
        name: "Ruthenium",
        mass: 101.1,
        covalent_radius: 1.46,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Rh",
        name: "Rhodium",
        mass: 102.9,
        covalent_radius: 1.42,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Pd",
        name: "Palladium",
        mass: 106.4,
        covalent_radius: 1.39,
        vdw_radius: 1.63,
    },
    ElementData {
        symbol: "Ag",
        name: "Silver",
        mass: 107.9,
        covalent_radius: 1.45,
        vdw_radius: 1.72,
    },
    ElementData {
        symbol: "Cd",
        name: "Cadmium",
        mass: 112.4,
        covalent_radius: 1.44,
        vdw_radius: 1.58,
    },
    ElementData {
        symbol: "In",
        name: "Indium",
        mass: 114.8,
        covalent_radius: 1.42,
        vdw_radius: 1.93,
    },
    ElementData {
        symbol: "Sn",
        name: "Tin",
        mass: 118.7,
        covalent_radius: 1.39,
        vdw_radius: 2.17,
    },
    ElementData {
        symbol: "Sb",
        name: "Antimony",
        mass: 121.8,
        covalent_radius: 1.39,
        vdw_radius: 2.06,
    },
    ElementData {
        symbol: "Te",
        name: "Tellurium",
        mass: 127.6,
        covalent_radius: 1.38,
        vdw_radius: 2.06,
    },
    ElementData {
        symbol: "I",
        name: "Iodine",
        mass: 126.9,
        covalent_radius: 1.39,
        vdw_radius: 1.98,
    },
    ElementData {
        symbol: "Xe",
        name: "Xenon",
        mass: 131.3,
        covalent_radius: 1.40,
        vdw_radius: 2.16,
    },
    ElementData {
        symbol: "Cs",
        name: "Cesium",
        mass: 132.9,
        covalent_radius: 2.44,
        vdw_radius: 3.43,
    },
    ElementData {
        symbol: "Ba",
        name: "Barium",
        mass: 137.3,
        covalent_radius: 2.15,
        vdw_radius: 2.68,
    },
    ElementData {
        symbol: "La",
        name: "Lanthanum",
        mass: 138.9,
        covalent_radius: 2.07,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ce",
        name: "Cerium",
        mass: 140.1,
        covalent_radius: 2.04,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Pr",
        name: "Praseodymium",
        mass: 140.9,
        covalent_radius: 2.03,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Nd",
        name: "Neodymium",
        mass: 144.2,
        covalent_radius: 2.01,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Pm",
        name: "Promethium",
        mass: 145.0,
        covalent_radius: 1.99,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Sm",
        name: "Samarium",
        mass: 150.4,
        covalent_radius: 1.98,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Eu",
        name: "Europium",
        mass: 152.0,
        covalent_radius: 1.98,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Gd",
        name: "Gadolinium",
        mass: 157.3,
        covalent_radius: 1.96,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Tb",
        name: "Terbium",
        mass: 158.9,
        covalent_radius: 1.94,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Dy",
        name: "Dysprosium",
        mass: 162.5,
        covalent_radius: 1.92,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ho",
        name: "Holmium",
        mass: 164.9,
        covalent_radius: 1.92,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Er",
        name: "Erbium",
        mass: 167.3,
        covalent_radius: 1.89,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Tm",
        name: "Thulium",
        mass: 168.9,
        covalent_radius: 1.90,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Yb",
        name: "Ytterbium",
        mass: 173.0,
        covalent_radius: 1.87,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Lu",
        name: "Lutetium",
        mass: 175.0,
        covalent_radius: 1.87,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Hf",
        name: "Hafnium",
        mass: 178.5,
        covalent_radius: 1.75,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ta",
        name: "Tantalum",
        mass: 180.9,
        covalent_radius: 1.70,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "W",
        name: "Tungsten",
        mass: 183.8,
        covalent_radius: 1.62,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Re",
        name: "Rhenium",
        mass: 186.2,
        covalent_radius: 1.51,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Os",
        name: "Osmium",
        mass: 190.2,
        covalent_radius: 1.44,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ir",
        name: "Iridium",
        mass: 192.2,
        covalent_radius: 1.41,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Pt",
        name: "Platinum",
        mass: 195.1,
        covalent_radius: 1.36,
        vdw_radius: 1.75,
    },
    ElementData {
        symbol: "Au",
        name: "Gold",
        mass: 197.0,
        covalent_radius: 1.36,
        vdw_radius: 1.66,
    },
    ElementData {
        symbol: "Hg",
        name: "Mercury",
        mass: 200.6,
        covalent_radius: 1.32,
        vdw_radius: 1.55,
    },
    ElementData {
        symbol: "Tl",
        name: "Thallium",
        mass: 204.4,
        covalent_radius: 1.45,
        vdw_radius: 1.96,
    },
    ElementData {
        symbol: "Pb",
        name: "Lead",
        mass: 207.2,
        covalent_radius: 1.46,
        vdw_radius: 2.02,
    },
    ElementData {
        symbol: "Bi",
        name: "Bismuth",
        mass: 209.0,
        covalent_radius: 1.48,
        vdw_radius: 2.07,
    },
    ElementData {
        symbol: "Po",
        name: "Polonium",
        mass: 209.0,
        covalent_radius: 1.40,
        vdw_radius: 1.97,
    },
    ElementData {
        symbol: "At",
        name: "Astatine",
        mass: 210.0,
        covalent_radius: 1.50,
        vdw_radius: 2.02,
    },
    ElementData {
        symbol: "Rn",
        name: "Radon",
        mass: 222.0,
        covalent_radius: 1.50,
        vdw_radius: 2.20,
    },
    ElementData {
        symbol: "Fr",
        name: "Francium",
        mass: 223.0,
        covalent_radius: 2.60,
        vdw_radius: 3.48,
    },
    ElementData {
        symbol: "Ra",
        name: "Radium",
        mass: 226.0,
        covalent_radius: 2.21,
        vdw_radius: 2.83,
    },
    ElementData {
        symbol: "Ac",
        name: "Actinium",
        mass: 227.0,
        covalent_radius: 2.15,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Th",
        name: "Thorium",
        mass: 232.0,
        covalent_radius: 2.06,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Pa",
        name: "Protactinium",
        mass: 231.0,
        covalent_radius: 2.00,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "U",
        name: "Uranium",
        mass: 238.0,
        covalent_radius: 1.96,
        vdw_radius: 1.86,
    },
    ElementData {
        symbol: "Np",
        name: "Neptunium",
        mass: 237.0,
        covalent_radius: 1.90,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Pu",
        name: "Plutonium",
        mass: 244.0,
        covalent_radius: 1.87,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Am",
        name: "Americium",
        mass: 243.0,
        covalent_radius: 1.80,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Cm",
        name: "Curium",
        mass: 247.0,
        covalent_radius: 1.69,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Bk",
        name: "Berkelium",
        mass: 247.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Cf",
        name: "Californium",
        mass: 251.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Es",
        name: "Einsteinium",
        mass: 252.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Fm",
        name: "Fermium",
        mass: 257.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Md",
        name: "Mendelevium",
        mass: 258.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "No",
        name: "Nobelium",
        mass: 259.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Lr",
        name: "Lawrencium",
        mass: 262.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Rf",
        name: "Rutherfordium",
        mass: 267.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Db",
        name: "Dubnium",
        mass: 268.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Sg",
        name: "Seaborgium",
        mass: 271.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Bh",
        name: "Bohrium",
        mass: 272.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Hs",
        name: "Hassium",
        mass: 270.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Mt",
        name: "Meitnerium",
        mass: 276.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ds",
        name: "Darmstadtium",
        mass: 281.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Rg",
        name: "Roentgenium",
        mass: 280.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Cn",
        name: "Copernicium",
        mass: 285.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Nh",
        name: "Nihonium",
        mass: 284.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Fl",
        name: "Flerovium",
        mass: 289.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Mc",
        name: "Moscovium",
        mass: 288.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Lv",
        name: "Livermorium",
        mass: 293.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Ts",
        name: "Tennessine",
        mass: 294.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
    ElementData {
        symbol: "Og",
        name: "Oganesson",
        mass: 294.0,
        covalent_radius: 1.60,
        vdw_radius: 2.00,
    },
];

impl Element {
    /// Atomic number (Z)
    #[inline]
    pub const fn z(self) -> u8 {
        self as u8
    }

    /// Chemical symbol (e.g., "H", "C", "Fe")
    #[inline]
    pub fn symbol(self) -> &'static str {
        ELEMENT_DATA[self.z() as usize - 1].symbol
    }

    /// English element name
    #[inline]
    pub fn name(self) -> &'static str {
        ELEMENT_DATA[self.z() as usize - 1].name
    }

    /// Standard atomic mass (in atomic mass units)
    #[inline]
    pub fn atomic_mass(self) -> f32 {
        ELEMENT_DATA[self.z() as usize - 1].mass
    }

    /// Covalent radius (in Angstroms)
    #[inline]
    pub fn covalent_radius(self) -> f32 {
        ELEMENT_DATA[self.z() as usize - 1].covalent_radius
    }

    /// Van der Waals radius (in Angstroms)
    #[inline]
    pub fn vdw_radius(self) -> f32 {
        ELEMENT_DATA[self.z() as usize - 1].vdw_radius
    }

    /// All supported elements (1-118)
    pub const ALL: &'static [Element] = &[
        Element::H,
        Element::He,
        Element::Li,
        Element::Be,
        Element::B,
        Element::C,
        Element::N,
        Element::O,
        Element::F,
        Element::Ne,
        Element::Na,
        Element::Mg,
        Element::Al,
        Element::Si,
        Element::P,
        Element::S,
        Element::Cl,
        Element::Ar,
        Element::K,
        Element::Ca,
        Element::Sc,
        Element::Ti,
        Element::V,
        Element::Cr,
        Element::Mn,
        Element::Fe,
        Element::Co,
        Element::Ni,
        Element::Cu,
        Element::Zn,
        Element::Ga,
        Element::Ge,
        Element::As,
        Element::Se,
        Element::Br,
        Element::Kr,
        Element::Rb,
        Element::Sr,
        Element::Y,
        Element::Zr,
        Element::Nb,
        Element::Mo,
        Element::Tc,
        Element::Ru,
        Element::Rh,
        Element::Pd,
        Element::Ag,
        Element::Cd,
        Element::In,
        Element::Sn,
        Element::Sb,
        Element::Te,
        Element::I,
        Element::Xe,
        Element::Cs,
        Element::Ba,
        Element::La,
        Element::Ce,
        Element::Pr,
        Element::Nd,
        Element::Pm,
        Element::Sm,
        Element::Eu,
        Element::Gd,
        Element::Tb,
        Element::Dy,
        Element::Ho,
        Element::Er,
        Element::Tm,
        Element::Yb,
        Element::Lu,
        Element::Hf,
        Element::Ta,
        Element::W,
        Element::Re,
        Element::Os,
        Element::Ir,
        Element::Pt,
        Element::Au,
        Element::Hg,
        Element::Tl,
        Element::Pb,
        Element::Bi,
        Element::Po,
        Element::At,
        Element::Rn,
        Element::Fr,
        Element::Ra,
        Element::Ac,
        Element::Th,
        Element::Pa,
        Element::U,
        Element::Np,
        Element::Pu,
        Element::Am,
        Element::Cm,
        Element::Bk,
        Element::Cf,
        Element::Es,
        Element::Fm,
        Element::Md,
        Element::No,
        Element::Lr,
        Element::Rf,
        Element::Db,
        Element::Sg,
        Element::Bh,
        Element::Hs,
        Element::Mt,
        Element::Ds,
        Element::Rg,
        Element::Cn,
        Element::Nh,
        Element::Fl,
        Element::Mc,
        Element::Lv,
        Element::Ts,
        Element::Og,
    ];

    /// Lookup by atomic number
    #[inline]
    pub fn by_number(z: u8) -> Option<Element> {
        if (1..=118).contains(&z) {
            Some(Self::ALL[(z - 1) as usize])
        } else {
            None
        }
    }

    /// Lookup by symbol (case-insensitive)
    pub fn by_symbol(sym: &str) -> Option<Element> {
        Self::ALL
            .iter()
            .copied()
            .find(|e| e.symbol().eq_ignore_ascii_case(sym))
    }

    /// Default allowed valences for this element (ascending order).
    ///
    /// Returns the list of typical valences used for implicit hydrogen
    /// calculation. The smallest value ≥ current bond-order sum is chosen.
    /// Returns `&[]` for noble gases and elements without standard valences.
    pub fn default_valences(self) -> &'static [u8] {
        match self {
            Element::H => &[1],
            Element::He => &[],
            Element::Li => &[1],
            Element::Be => &[2],
            Element::B => &[3],
            Element::C => &[4],
            Element::N => &[3, 5],
            Element::O => &[2],
            Element::F => &[1],
            Element::Ne => &[],
            Element::Na => &[1],
            Element::Mg => &[2],
            Element::Al => &[3],
            Element::Si => &[4],
            Element::P => &[3, 5],
            Element::S => &[2, 4, 6],
            Element::Cl => &[1, 3, 5, 7],
            Element::Ar => &[],
            Element::K => &[1],
            Element::Ca => &[2],
            Element::Sc => &[3],
            Element::Ti => &[2, 3, 4],
            Element::V => &[2, 3, 4, 5],
            Element::Cr => &[2, 3, 6],
            Element::Mn => &[2, 3, 4, 6, 7],
            Element::Fe => &[2, 3],
            Element::Co => &[2, 3],
            Element::Ni => &[2, 3],
            Element::Cu => &[1, 2],
            Element::Zn => &[2],
            Element::Ga => &[3],
            Element::Ge => &[2, 4],
            Element::As => &[3, 5],
            Element::Se => &[2, 4, 6],
            Element::Br => &[1, 3, 5, 7],
            Element::Kr => &[],
            Element::Rb => &[1],
            Element::Sr => &[2],
            Element::Y => &[3],
            Element::Zr => &[4],
            Element::Nb => &[3, 5],
            Element::Mo => &[2, 3, 4, 5, 6],
            Element::Tc => &[4, 6, 7],
            Element::Ru => &[2, 3, 4, 6, 8],
            Element::Rh => &[2, 3],
            Element::Pd => &[2, 4],
            Element::Ag => &[1],
            Element::Cd => &[2],
            Element::In => &[3],
            Element::Sn => &[2, 4],
            Element::Sb => &[3, 5],
            Element::Te => &[2, 4, 6],
            Element::I => &[1, 3, 5, 7],
            Element::Xe => &[],
            Element::Cs => &[1],
            Element::Ba => &[2],
            Element::La => &[3],
            Element::Ce => &[3, 4],
            Element::Pr => &[3, 4],
            Element::Nd => &[3],
            Element::Pm => &[3],
            Element::Sm => &[2, 3],
            Element::Eu => &[2, 3],
            Element::Gd => &[3],
            Element::Tb => &[3, 4],
            Element::Dy => &[3],
            Element::Ho => &[3],
            Element::Er => &[3],
            Element::Tm => &[2, 3],
            Element::Yb => &[2, 3],
            Element::Lu => &[3],
            Element::Hf => &[4],
            Element::Ta => &[5],
            Element::W => &[2, 3, 4, 5, 6],
            Element::Re => &[2, 4, 6, 7],
            Element::Os => &[2, 3, 4, 6, 8],
            Element::Ir => &[2, 3, 4],
            Element::Pt => &[2, 4],
            Element::Au => &[1, 3],
            Element::Hg => &[1, 2],
            Element::Tl => &[1, 3],
            Element::Pb => &[2, 4],
            Element::Bi => &[3, 5],
            Element::Po => &[2, 4],
            Element::At => &[1],
            Element::Rn => &[],
            Element::Fr => &[1],
            Element::Ra => &[2],
            Element::Ac => &[3],
            Element::Th => &[4],
            Element::Pa => &[4, 5],
            Element::U => &[3, 4, 5, 6],
            Element::Np => &[3, 4, 5, 6],
            Element::Pu => &[3, 4, 5, 6],
            Element::Am => &[2, 3, 4, 5, 6],
            Element::Cm => &[3],
            Element::Bk => &[3, 4],
            Element::Cf => &[2, 3, 4],
            Element::Es => &[2, 3],
            Element::Fm => &[2, 3],
            Element::Md => &[2, 3],
            Element::No => &[2, 3],
            Element::Lr => &[3],
            // Transactinides and superheavy – no meaningful organic valence
            _ => &[],
        }
    }
}

impl FromStr for Element {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Element::by_symbol(s).ok_or(())
    }
}

impl core::fmt::Display for Element {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.symbol())
    }
}

#[cfg(test)]
mod test_element {
    use super::*;

    #[test]
    fn test_accessors() {
        assert_eq!(Element::by_number(1), Some(Element::H));
        assert_eq!(Element::by_symbol("H"), Some(Element::H));
        assert_eq!(Element::by_symbol("h"), Some(Element::H));
        assert_eq!(Element::by_symbol("X"), None);
        assert_eq!(Element::by_number(0), None);
        assert_eq!(Element::by_number(119), None);
    }

    #[test]
    fn test_props() {
        let e = Element::H;
        assert_eq!(e.z(), 1);
        assert_eq!(e.symbol(), "H");
        assert_eq!(e.name(), "Hydrogen");
        assert_eq!(e.atomic_mass(), 1.008);
        assert_eq!(format!("{}", e), "H");
    }

    #[test]
    fn test_common_elements() {
        let c = Element::C;
        assert_eq!(c.z(), 6);
        assert_eq!(c.symbol(), "C");
        assert_eq!(c.name(), "Carbon");
        assert!((c.atomic_mass() - 12.01).abs() < 0.01);

        let fe = Element::Fe;
        assert_eq!(fe.z(), 26);
        assert_eq!(fe.symbol(), "Fe");
        assert_eq!(fe.name(), "Iron");

        let au = Element::Au;
        assert_eq!(au.z(), 79);
        assert_eq!(au.symbol(), "Au");
        assert_eq!(au.name(), "Gold");
    }

    #[test]
    fn test_new_properties() {
        let c = Element::C;
        assert!((c.covalent_radius() - 0.76).abs() < 0.01);
        assert!((c.vdw_radius() - 1.70).abs() < 0.01);

        let h = Element::H;
        assert!((h.covalent_radius() - 0.31).abs() < 0.01);
        assert!((h.vdw_radius() - 1.20).abs() < 0.01);
    }

    #[test]
    fn test_all_elements() {
        assert_eq!(Element::ALL.len(), 118);

        // Test first and last
        assert_eq!(Element::ALL[0], Element::H);
        assert_eq!(Element::ALL[117], Element::Og);

        // Test all elements have valid z
        for (i, elem) in Element::ALL.iter().enumerate() {
            assert_eq!(elem.z() as usize, i + 1);
        }
    }

    #[test]
    fn test_symbol_lookup() {
        assert_eq!(Element::by_symbol("C"), Some(Element::C));
        assert_eq!(Element::by_symbol("c"), Some(Element::C));
        assert_eq!(Element::by_symbol("Fe"), Some(Element::Fe));
        assert_eq!(Element::by_symbol("fe"), Some(Element::Fe));
        assert_eq!(Element::by_symbol("FE"), Some(Element::Fe));
        assert_eq!(Element::by_symbol("Og"), Some(Element::Og));
    }
}
