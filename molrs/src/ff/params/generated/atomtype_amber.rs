//! Antechamber atom-type definition rules — `ATOMTYPE_AMBER.DEF`.
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`.
//!
//! Source: `$AMBERHOME/dat/antechamber/ATOMTYPE_AMBER.DEF` (AmberTools).

use crate::ff::params::{
    AtdRule, AtdTable, AtomPattern, AtomProp, PatternAtom, PropConstraint, PropExpr, PropUnit,
    WildAtom, WildAtomSpec,
};

/// `WILDATOM XX C N O S P`
#[rustfmt::skip]
const WILD_XX: &[WildAtomSpec] = &[
    WildAtomSpec { z: 6, degree: None },
    WildAtomSpec { z: 7, degree: None },
    WildAtomSpec { z: 8, degree: None },
    WildAtomSpec { z: 16, degree: None },
    WildAtomSpec { z: 15, degree: None },
];

/// `WILDATOM XA O S`
#[rustfmt::skip]
const WILD_XA: &[WildAtomSpec] = &[
    WildAtomSpec { z: 8, degree: None },
    WildAtomSpec { z: 16, degree: None },
];

/// `WILDATOM XB N P`
#[rustfmt::skip]
const WILD_XB: &[WildAtomSpec] = &[
    WildAtomSpec { z: 7, degree: None },
    WildAtomSpec { z: 15, degree: None },
];

/// `WILDATOM XC F Cl Br I`
#[rustfmt::skip]
const WILD_XC: &[WildAtomSpec] = &[
    WildAtomSpec { z: 9, degree: None },
    WildAtomSpec { z: 17, degree: None },
    WildAtomSpec { z: 35, degree: None },
    WildAtomSpec { z: 53, degree: None },
];

/// The 4 `WILDATOM` aliases declared by `ATOMTYPE_AMBER.DEF`.
#[rustfmt::skip]
pub const WILDATOMS: &[WildAtom] = &[
    WildAtom { name: "XX", specs: WILD_XX },
    WildAtom { name: "XA", specs: WILD_XA },
    WildAtom { name: "XB", specs: WILD_XB },
    WildAtom { name: "XC", specs: WILD_XC },
];

/// The 64 `ATD` rules of `ATOMTYPE_AMBER.DEF`, in file order.
///
/// Order is significant: the FIRST rule that matches wins. `DU` (dummy) rows and
/// the residue-less catch-all carry no constraints and are not emitted.
pub const RULES: &[AtdRule] = &[
    AtdRule {
        atom_type: "CT",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "C",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XA),
            degree: Some(1),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CN",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(1),
                    degree: None,
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CB",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CR",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CR",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(1),
                    degree: None,
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CK",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CC",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: Some(0),
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[
                    AtomPattern {
                        atom: PatternAtom::Element(6),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[],
                    },
                    AtomPattern {
                        atom: PatternAtom::Element(1),
                        degree: None,
                        property: None,
                        label: None,
                        children: &[],
                    },
                ],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CC",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: Some(0),
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(6),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CW",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(1),
                    degree: None,
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CV",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "C*",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CQ",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(2),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(6),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(7),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(6),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(6),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(7),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(6),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(6),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(7),
                        degree: Some(2),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(6),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(6),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(6),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(7),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(6),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(7),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CA",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar1,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    },
                ],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CA",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CD",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CZ",
        residue: "*",
        atomic_number: Some(6),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HO",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(8),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HS",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(16),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HP",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: None,
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(4),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HW",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(8),
            degree: None,
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(1),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H3",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(3),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H2",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(2),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H1",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(1),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HC",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H5",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(2),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H4",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(1),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HA",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HA",
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "F",
        residue: "*",
        atomic_number: Some(9),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cl",
        residue: "*",
        atomic_number: Some(17),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Br",
        residue: "*",
        atomic_number: Some(35),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "I",
        residue: "*",
        atomic_number: Some(53),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "P",
        residue: "*",
        atomic_number: Some(15),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N1",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NB",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NC",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N2",
        residue: "*",
        atomic_number: Some(7),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
            ],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N2",
        residue: "*",
        atomic_number: Some(7),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
            ],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NO",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(8),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(8),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NA",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Rg5,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Rg6,
                            relation: None,
                        },
                    ],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N2",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Nr,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N*",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar1,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    },
                ],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XA),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NT",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N3",
        residue: "*",
        atomic_number: Some(7),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O2",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: None,
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(8),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O2",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(15),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(16),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "OH",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "OW",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: Some(2),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "OS",
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "SH",
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "SH",
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: Some(2),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "S",
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "SO",
        residue: "*",
        atomic_number: Some(16),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "LP",
        residue: "*",
        atomic_number: Some(0),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
];

/// `ATOMTYPE_AMBER.DEF` as one typed table.
pub const ATOMTYPE_AMBER: AtdTable = AtdTable {
    name: "ATOMTYPE_AMBER.DEF",
    wildatoms: WILDATOMS,
    rules: RULES,
};
