import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

N = 10_000

# First Gaussian cluster (centered at (0,0))
x1 = np.random.normal(loc=0.0, scale=0.05, size=N)
y1 = np.random.normal(loc=0.0, scale=0.05, size=N)

# Second Gaussian cluster (centered at (1,1))
x2 = np.random.normal(loc=0.15, scale=0.05, size=N)
y2 = np.random.normal(loc=0.2, scale=0.05, size=N)

# Concatenate to make a bi-modal distribution
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
""" Color Palettes with Linestyle

Here you go — neatly formatted in **Markdown** for quick reference:

---

## 🎨 Color Palettes & Linestyles

### **Photons (top panels)**

| Component | Color  | Hex Code  | Linestyle |
| --------- | ------ | --------- | --------- |
| BS($e^-$) | Orange | `#E69F00` | Solid     |
| π$^0$     | Blue   | `#0072B2` | Solid     |
| BS($e^+$) | Green  | `#009E73` | Solid     |
| Total     | Black  | `#000000` | Dashed    |

---

### **Electrons / Positrons (bottom panels)**

| Component       | Color         | Hex Code  | Linestyle |
| --------------- | ------------- | --------- | --------- |
| CR $e^-$        | Red           | `#D55E00` | Solid     |
| $e^-$(CR p)     | Orange        | `#E69F00` | Solid     |
| $e^-$(CR $e^-$) | Yellow-Orange | `#F0E442` | Dashed    |
| $e^+$(pair)     | Green         | `#009E73` | Dotted    |
| $e^+$(π$^+$)    | Green         | `#009E73` | Dash–dot  |
| $e^-$(π$^-$)    | Green         | `#009E73` | Solid     |
| Total           | Black         | `#000000` | Dashed    |

---

Would you like me to also build a **ready-to-use Matplotlib dictionary** (e.g. `styles = {"CR e-": {"color": ..., "linestyle": ...}, ...}`) so you can plug it directly into your plotting scripts?


"""

styles = {
    # --- Photons ---
    "BS(e-)":     {"color": "#E69F00", "linestyle": "-"},
    "pi0":        {"color": "#0072B2", "linestyle": "-"},
    "BS(e+)":     {"color": "#009E73", "linestyle": "-"},
    "total_ph":   {"color": "#000000", "linestyle": "--"},

    # --- Electrons / Positrons ---
    "CR e-":      {"color": "#D55E00", "linestyle": "-"},
    "e-(CR p)":   {"color": "#E69F00", "linestyle": "-"},
    "e-(CR e-)":  {"color": "#F0E442", "linestyle": "--"},
    "e+(pair)":   {"color": "#009E73", "linestyle": ":"},
    "e+(pi+)":    {"color": "#009E73", "linestyle": "-."},
    "e-(pi-)":    {"color": "#009E73", "linestyle": "-"},
    "total_e":    {"color": "#000000", "linestyle": "--"},
}

#label = "CR e-"
#plt.scatter(x, y, label=label, **styles[label])
#plt.savefig('./C.png')

color_palettes = {
    "tab10": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ],
    "tab20": [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5"
    ],
    "pastel": [
        "#aec6cf", "#ffb347", "#77dd77", "#ff6961", "#cbaacb",
        "#fdfd96", "#84b6f4", "#fdcae1", "#b39eb5", "#ffb3ba"
    ],
    "bright": [
        "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
        "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080"
    ],
    "dark": [
        "#2d2d2d", "#b22222", "#228b22", "#1e90ff", "#8b008b",
        "#ff8c00", "#708090", "#556b2f", "#4682b4", "#8b4513",
        "#9932cc", "#b8860b", "#2f4f4f", "#696969", "#191970",
        "#cd5c5c", "#6b8e23", "#20b2aa", "#b0c4de", "#d2691e"
    ],

    # --- ColorBrewer sets ---
    "Set1": [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"
    ],
    "Set2": [
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"
    ],
    "Paired": [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
        "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
        "#ffff99", "#b15928"
    ],
    "Accent": [
        "#7fc97f", "#beaed4", "#fdc086", "#ffff99",
        "#386cb0", "#f0027f", "#bf5b17", "#666666"
    ],

    # --- Scientific color maps (Fabio Crameri) ---
    "batlow": [
        "#011959", "#22408a", "#3e658f", "#5d7d77", "#7f9161",
        "#a19f57", "#c2ad5e", "#e2be6a", "#f8d082", "#fdf5d7"
    ],
    "roma": [
        "#3b0f70", "#8c2981", "#de4968", "#f89441", "#f0f921"
    ],
    "vik": [
        "#00204d", "#005792", "#0085a1", "#02a6a8", "#53c6c9",
        "#b0e0e6", "#f4e5cf", "#f0cfa1", "#ec9b72", "#d64c55",
        "#7f0f4c"
    ],
    "devon": [
        "#2e1a47", "#573c76", "#7a5997", "#9f77b7", "#c195d6",
        "#d6b1ea", "#ead3f7", "#f6efff"
    ],

    # --- Extra palettes ---
    "tol_bright": [
        "#EE6677", "#228833", "#4477AA", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB"
    ],
    "tol_muted": [
        "#88CCEE", "#44AA99", "#117733", "#332288",
        "#DDCC77", "#999933", "#CC6677", "#882255", "#AA4499"
    ],
    "cubehelix": [
        "#000000", "#1a1530", "#313682", "#1a7eb7", "#2bb07f",
        "#92c857", "#ead61c", "#fba60a", "#f44d2d", "#d31161", "#8e0d75"
    ],
    "rainbow": [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999",
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#000000"
    ]
}

# --- Dummy data ---
x = np.linspace(0, 10, 200)           # x-axis
ys = [np.sin(x + phase) for phase in np.linspace(0, 2*np.pi, 8)]  # 8 curves

# --- Example palette (choose any from color_palettes) ---
palette = color_palettes["Set1"]

# --- Plotting ---
plt.figure(figsize=(8,5))
for i, y in enumerate(ys):
    plt.plot(x, y, color=palette[i % len(palette)], label=f"curve {i+1}")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Example with color palette")
plt.savefig('./C.png')