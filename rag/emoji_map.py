"""Discord emoji shortcode -> human-readable RS3 name.

Shortcodes listed in ALIAS_CODES are rendered as "Full Name (code)" rather than
just "Full Name". These are the abbreviations players actually type - someone
asks "when do I use fsoa?", not "when do I use the Fractured Staff of Armadyl?".
Expanding them away would delete the exact token the query depends on, and the
abbreviation is a rare high-signal term for BM25 once hybrid search lands.

It also limits the damage from a wrong mapping: the original code survives.

The guides are 30-39% emoji markup by character count. The numeric IDs are pure
noise to an embedding model, so every shortcode is expanded to the name a
player would actually type in a question.
"""

EMOJI_MAP: dict[str, str] = {
    # --- Necromancy abilities ---
    "invokedeath": "Invoke Death",
    "livingdeath": "Living Death",
    "touchofdeath": "Touch of Death",
    "fingerofdeath": "Finger of Death",
    "deathskulls": "Death Skulls",
    "soulsap": "Soul Sap",
    "soulstrikeflank": "Soul Strike (flanking)",
    "volleyofsouls": "Volley of Souls",
    "spectralscythe": "Spectral Scythe",
    "spectralscythe2": "Spectral Scythe (2nd hit)",
    "spectralscythe3": "Spectral Scythe (3rd hit)",
    "threadsoffate": "Threads of Fate",
    "splitsoul": "Split Soul",
    "bloat": "Bloat",
    "lifetransfer": "Life Transfer",
    "darkness": "Darkness",
    # --- Conjures ---
    "conjurearmy": "Conjure Undead Army",
    "commandskeleton": "Command Skeleton Warrior",
    "commandghost": "Command Vengeful Ghost",
    "commandzombie": "Command Putrid Zombie",
    "commandphantom": "Command Phantom Guardian",
    # --- Resources ---
    "necrosis": "Necrosis stack",
    "residualsoul": "Residual Soul",
    "necromancy": "Necromancy",
    # --- Defensives / utility ---
    "devo": "Devotion",            # abbrev
    "cade": "Barricade",           # abbrev
    "debil": "Debilitate",         # abbrev
    "anti": "Anticipation",        # abbrev - blocks stuns, not anti-fire
    "res": "Resonance",            # abbrev
    "reflect": "Reflect",
    "reprisal": "Reprisal",
    "disrupt": "Disruption Shield",
    "shielddome": "Shield Dome",
    "freedom": "Freedom",
    "prep": "Preparation",
    "veng": "Vengeance",           # abbrev
    "divert": "Divert",
    "soulsplit": "Soul Split",
    "adrenrenewal": "Adrenaline Renewal",
    "powerburstofvitality": "Powerburst of Vitality",
    "limitless": "Limitless Sigil",
    # --- Weapons / gear ---
    "deathguard90": "Death Guard",
    "omniguard": "Omni Guard",
    "excal": "Augmented Excalibur",  # abbrev
    "eofspec": "Essence of Finality special attack",
    "spec": "special attack",
    "roarofawakening": "Roar of Awakening",
    "odetodeceit": "Ode to Deceit",
    # --- Consumables / buffs ---
    "vulnbomb": "Vulnerability bomb",
    "weppoison": "weapon poison",
    "powderofpenance": "Powder of Penance",
    "thermalflask": "thermal spa flask",
    "kwuarmsticks": "Kwuarm incense sticks",
    # --- Familiars ---
    "ripperpouch": "Ripper Demon pouch",
    "hellhoundpouch": "Hellhound pouch",
    "kalgpouch": "Kal'gerion Demon pouch",
    # ------------------------------------------------------------------
    # Added for the Melee/Magic and Melee/Ranged hybrid guides
    # ------------------------------------------------------------------
    # --- Magic abilities ---
    "gsonic": "Greater Sonic Wave",
    "gconc": "Greater Concentrated Blast",
    "gsunshine": "Greater Sunshine",
    "gchain": "Greater Chain",
    "wm": "Wild Magic",
    "asphyx": "Asphyxiate",
    "tsunami": "Tsunami",
    "dbreath": "Dragon Breath",
    "magmatemptest": "Magma Tempest",
    "meteorstrike": "Meteor Strike",
    "bloodbarrage": "Blood Barrage",
    "smokecloud": "Smoke Cloud",
    "smoketendrils": "Smoke Tendrils",
    "shadowtend": "Shadow Tendrils",
    "imbueshadows": "Imbue Shadows",
    "adaptivestrike": "Adaptive Strike",
    "omni": "Omnipower",
    "runic_charge": "Runic Charge",
    # --- Ranged abilities ---
    "gdeathsswift": "Greater Death's Swiftness",
    "grico": "Greater Ricochet",
    "deadshot": "Deadshot",
    "snapshot": "Snapshot",
    "snipe": "Snipe",
    "rapid": "Rapid Fire",
    "piercingshot": "Piercing Shot",
    "galeshot": "Gale Shot",
    "deathsporearrows": "Death Spore Arrows",
    "decimation": "Decimation",
    # --- Melee abilities ---
    "berserk": "Berserk",
    "gbarge": "Greater Barge",
    "gflurry": "Greater Flurry",
    "assault": "Assault",
    "overpower": "Overpower",
    "punish": "Punish",
    "rend": "Rend",
    "pulverise": "Pulverise",
    "chaosroar": "Chaos Roar",
    "cane": "Hurricane",  # shortcode is the tail of 'hurriCANE'
    "bloodlust": "Bloodlust",
    "varanussmercy": "Varanus's Mercy",
    # --- Debuffs / utility ---
    "enfeeble": "Enfeeble",
    "caroming4": "Caroming 4",
    "DeflectMage": "Deflect Magic",
    "prismofrestoration": "Prism of Restoration",
    "powderofprotection": "Powder of Protection",
    # --- Weapons / gear ---
    "fsoa": "Fractured Staff of Armadyl",
    "bolg": "Bow of the Last Guardian",
    "ecb": "Eldritch Crossbow",
    "sgb": "Seren godbow",
    "gloomfirebow": "Gloomfire bow",
    "dba": "Dragon battleaxe",
    "ezk": "Ek-ZekKil",
    "amhej": "Am-hej",
    "magiccape": "Master Magic Cape",
    "tumekenslight": "Tumeken's Light",
    "zemouregalsnexus": "Zemouregal's Nexus",
    "nodonspikeharness": "Nodon spike harness",
    # --- Combat style labels (decorative, next to preset links) ---
    "melee": "Melee",
    "magic": "Magic",
    "range": "Ranged",
}

# Rendered as "Full Name (code)" so both the name and the jargon are searchable.
ALIAS_CODES: set[str] = {
    # Weapons
    "fsoa", "bolg", "ecb", "sgb", "dba", "ezk", "amhej",
    # Magic abilities
    "gconc", "gsonic", "gsunshine", "gchain", "wm", "asphyx", "dbreath", "omni",
    # Ranged abilities
    "grico", "gdeathsswift",
    # Melee abilities
    "gbarge", "gflurry", "cane",
    # Defensives / utility
    "devo", "cade", "debil", "anti", "res", "veng", "prep", "disrupt", "excal",
    # Specs
    "eofspec", "vulnbomb", "DeflectMage",
}
