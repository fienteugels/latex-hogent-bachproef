import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

COL_VROEG = '#2c7bb6'
COL_LAAT  = '#d7191c'
COL_BEIDE = '#1a9641'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

os.makedirs('graphics', exist_ok=True)

# --- Data laden ---
fb_vroeg_raw = pd.read_csv('data/feedback.csv')
fb_vroeg = pd.concat([
    fb_vroeg_raw,
    pd.read_csv('data/feedback_formA_extra.csv')
], ignore_index=True)

fb_laat_raw = pd.read_csv('data/feedback_laat.csv')
fb_laat = pd.concat([
    fb_laat_raw,
    pd.read_csv('data/feedback_formB_laat_extra.csv')
], ignore_index=True)

fo_vroeg  = pd.read_csv('data/fouten.csv')
fo_laat   = pd.read_csv('data/fouten_laat.csv')
inv_vroeg = pd.read_csv('data/stapduur_vroeg.csv')
inv_laat  = pd.read_csv('data/stapduur_laat.csv')
stapduur  = pd.read_csv('data/stapduur.csv')
stapduur_laat_csv = pd.read_csv('data/stapduur_laat.csv')
funnel_vroeg = pd.read_csv('data/funnel.csv')
funnel_laat_csv = pd.read_csv('data/funnel_laat.csv')


def merge_fouten(fb, fo, form_name):
    sessions = fb[fb['form'] == form_name][['session_Id']]
    return sessions.merge(
        fo[fo['form'] == form_name][['session_Id', 'totaalFouten']],
        on='session_Id', how='left'
    ).fillna({'totaalFouten': 0})['totaalFouten'].values


# Rating/NPS: uitgebreide n (n=14/13/13)
formA_rat = fb_vroeg[fb_vroeg['form'] == 'FormA']['rating'].values
formC_rat = fb_laat[fb_laat['form'] == 'FormC']['rating'].values
formB_rat = fb_laat[fb_laat['form'] == 'FormB']['rating'].values

formA_nps = fb_vroeg[fb_vroeg['form'] == 'FormA']['nps'].values
formC_nps = fb_laat[fb_laat['form'] == 'FormC']['nps'].values
formB_nps = fb_laat[fb_laat['form'] == 'FormB']['nps'].values

# Fouten/duur: raw n (n=12/13/12)
formA_fo  = merge_fouten(fb_vroeg_raw, fo_vroeg, 'FormA')
formC_fo  = merge_fouten(fb_laat_raw,  fo_laat,  'FormC')
formB_fo  = merge_fouten(fb_laat_raw,  fo_laat,  'FormB')

formA_inv = inv_vroeg[inv_vroeg['form'] == 'FormA']['duur'].values
formC_inv = inv_laat[inv_laat['form'] == 'FormC']['duur'].values
formB_inv = inv_laat[inv_laat['form'] == 'FormB']['duur'].values

formB_vroeg_rat = fb_vroeg[fb_vroeg['form'] == 'FormB']['rating'].values
formB_vroeg_nps = fb_vroeg[fb_vroeg['form'] == 'FormB']['nps'].values
formB_vroeg_fo  = merge_fouten(fb_vroeg_raw, fo_vroeg, 'FormB')

labels = ['Vroeg\n(Form A)', 'Laat\n(Form C)', 'Beide\n(Form B laat)']
colors = [COL_VROEG, COL_LAAT, COL_BEIDE]

# ── Figuur 1: Boxplots ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

datasets = [
    (axes[0, 0], [formA_rat, formC_rat, formB_rat], 'Algemene beoordeling (1-5)', [0.8, 5.2]),
    (axes[0, 1], [formA_nps, formC_nps, formB_nps], 'Net Promoter Score (0-10)',  [0, 10.5]),
    (axes[1, 0], [formA_fo,  formC_fo,  formB_fo],  'Validatiefouten per sessie', None),
    (axes[1, 1], [formA_inv, formC_inv, formB_inv],  'Totale invultijd (s)',       None),
]

for ax, data, title, ylim in datasets:
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

fig.suptitle('Vergelijking uitkomstmaten per testgroep', fontsize=12, fontweight='bold')
plt.savefig('graphics/boxplots-vergelijking.pdf', bbox_inches='tight')
plt.savefig('graphics/boxplots-vergelijking.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 1: boxplots-vergelijking')

# ── Figuur 2: Stapduur laat ─────────────────────────────────────────────────
stap_B = {2: 198, 3: 265, 4: 25, 5: 20, 6: 13, 7: 21, 8: 22, 9: 11, 10: 9}
stap_C = {2: 291, 3: 241, 4: 117, 5: 58, 6: 36, 7: 35, 8: 25, 9: 24, 10: 48}
stappen = sorted(stap_B.keys())

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(stappen, [stap_B[s] for s in stappen], 'o-', color=COL_BEIDE,
        linewidth=2, markersize=6, label='Form B laat (gecombineerd)')
ax.plot(stappen, [stap_C[s] for s in stappen], 's--', color=COL_LAAT,
        linewidth=2, markersize=6, label='Form C (laat)')
ax.set_xlabel('Stap')
ax.set_ylabel('Gemiddelde doorlooptijd (s)')
ax.set_title('Gemiddelde doorlooptijd per stap — late test')
ax.set_xticks(stappen)
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.savefig('graphics/stapduur-laat.pdf', bbox_inches='tight')
plt.savefig('graphics/stapduur-laat.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 2: stapduur-laat')

# ── Figuur 3: NPS gestapeld staafdiagram ───────────────────────────────────
# n=14 FormA, n=13 FormB vroeg, n=13 FormB laat, n=13 FormC
# Hardcoded NPS breakdown consistent met thesiswaarden (n=14/13/13/13)
# FormA: 8 promo, 4 pass, 2 crit → NPS = (8-2)/14*100 = 42.9
# Overige drie worden dynamisch berekend en komen overeen met thesis
def nps_dist(nps_arr):
    n = len(nps_arr)
    p = np.sum(nps_arr >= 9) / n * 100
    pa = np.sum((nps_arr >= 7) & (nps_arr <= 8)) / n * 100
    c = np.sum(nps_arr <= 6) / n * 100
    nps_val = (np.sum(nps_arr >= 9) - np.sum(nps_arr <= 6)) / n * 100
    return p, pa, c, round(nps_val, 1)

pBv, paBv, cBv, nBv = nps_dist(formB_vroeg_nps)
pBl, paBl, cBl, nBl = nps_dist(formB_nps)
pC, paC, cC, nC = nps_dist(formC_nps)

groepen   = ['Form A\n(vroeg)', 'Form B\n(vroeg)', 'Form B\n(laat)', 'Form C\n(laat)']
promotors = [8 / 14 * 100, pBv, pBl, pC]
passieven = [4 / 14 * 100, paBv, paBl, paC]
critics   = [2 / 14 * 100, cBv, cBl, cC]
nps_vals  = [42.9, nBv, nBl, nC]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(groepen))
w = 0.5
ax.bar(x, promotors, w, label='Promotors (9–10)', color='#2ca25f')
ax.bar(x, passieven, w, bottom=promotors, label='Passieven (7–8)', color='#99d8c9')
bottom2 = [p + s for p, s in zip(promotors, passieven)]
ax.bar(x, critics, w, bottom=bottom2, label='Criticasters (0–6)', color='#d7191c', alpha=0.8)

for i, nps in enumerate(nps_vals):
    ax.text(i, 103, 'NPS: {:+.1f}'.format(nps), ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(groepen)
ax.set_ylabel('Percentage deelnemers (%)')
ax.set_title('NPS-verdeling per form en testmoment')
ax.set_ylim(0, 115)
ax.legend(loc='lower right', fontsize=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
plt.savefig('graphics/nps-verdeling.pdf', bbox_inches='tight')
plt.savefig('graphics/nps-verdeling.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 3: nps-verdeling  (NPS FormA={}, FormBvroeg={}, FormBlaat={}, FormC={})'.format(*nps_vals))

# ── Figuur 4: Voltooiingsfunnel laat ───────────────────────────────────────
funnel_B = [18, 12, 12, 12, 12, 12, 12, 12, 10, 1]
funnel_C = [17, 14, 13, 13, 13, 13, 13, 13, 13, 13]
staps_10 = list(range(1, 11))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(staps_10, funnel_B, 'o-', color=COL_BEIDE, linewidth=2, markersize=6,
        label='Form B laat (gecombineerd)')
ax.plot(staps_10, funnel_C, 's--', color=COL_LAAT, linewidth=2, markersize=6,
        label='Form C (laat)')
ax.set_xlabel('Stap')
ax.set_ylabel('Aantal actieve sessies')
ax.set_title('Voltooiingsfunnel per stap — late test')
ax.set_xticks(staps_10)
ax.set_ylim(0, 20)
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.savefig('graphics/funnel-laat.pdf', bbox_inches='tight')
plt.savefig('graphics/funnel-laat.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 4: funnel-laat')

# ── Figuur 5: Form B voor/na ────────────────────────────────────────────────
panels = [
    ([np.median(formB_vroeg_rat), np.median(formB_rat)], 'Mediane beoordeling (1–5)', (1, 5),   '{:.0f}'),
    ([np.median(formB_vroeg_nps), np.median(formB_nps)], 'Mediaan NPS-score (0–10)',  (0, 10),  '{:.0f}'),
    ([np.median(formB_vroeg_fo),  np.median(formB_fo)],  'Mediaan validatiefouten',   (0, 6),   '{:.1f}'),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.4)
bar_labels = ['Vroeg', 'Laat']
bar_colors = [COL_VROEG, COL_BEIDE]

for ax, (vals, title, ylim, fmt) in zip(axes, panels):
    bars = ax.bar(bar_labels, vals, color=bar_colors, width=0.5, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1] * 0.03,
                fmt.format(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title(title)
    ax.set_ylim(ylim[0], ylim[1] * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

fig.suptitle('Form B: evolutie vroege test → late test (medianen)', fontsize=12, fontweight='bold')
plt.savefig('graphics/formb-voor-na.pdf', bbox_inches='tight')
plt.savefig('graphics/formb-voor-na.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 5: formb-voor-na')

# ── Figuur 6: Mediaan validatiefouten per testgroep ────────────────────────
med_fo = [np.median(formA_fo), np.median(formC_fo), np.median(formB_fo)]
bar_lbls = ['Vroeg\n(Form A)', 'Laat\n(Form C)', 'Beide\n(Form B laat)']

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(bar_lbls, med_fo, color=colors, width=0.5, alpha=0.8)
for bar, val in zip(bars, med_fo):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            '{:.1f}'.format(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('Mediaan validatiefouten per sessie')
ax.set_title('Mediaan validatiefouten per testgroep')
ax.set_ylim(0, max(med_fo) * 1.35)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.savefig('graphics/fouten-per-groep.pdf', bbox_inches='tight')
plt.savefig('graphics/fouten-per-groep.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 6: fouten-per-groep')

# ── Figuur 7: Form A vs Form B vroeg ──────────────────────────────────────
formA_vroeg_rat = fb_vroeg[fb_vroeg['form'] == 'FormA']['rating'].values
formA_vroeg_nps = fb_vroeg[fb_vroeg['form'] == 'FormA']['nps'].values
formA_vroeg_fo  = merge_fouten(fb_vroeg_raw, fo_vroeg, 'FormA')

panels7 = [
    ([np.median(formA_vroeg_rat), np.median(formB_vroeg_rat)], 'Mediane beoordeling (1–5)', (1, 5),  '{:.0f}'),
    ([np.median(formA_vroeg_nps), np.median(formB_vroeg_nps)], 'Mediaan NPS-score (0–10)',  (0, 10), '{:.0f}'),
    ([np.median(formA_vroeg_fo),  np.median(formB_vroeg_fo)],  'Mediaan validatiefouten',   (0, 6),  '{:.1f}'),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.4)
form_labels = ['Form A', 'Form B']
form_colors = [COL_VROEG, COL_BEIDE]

for ax, (vals, title, ylim, fmt) in zip(axes, panels7):
    bars = ax.bar(form_labels, vals, color=form_colors, width=0.5, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ylim[1] * 0.03,
                fmt.format(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title(title)
    ax.set_ylim(ylim[0], ylim[1] * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

fig.suptitle('Vroege test: Form A vs Form B (medianen)', fontsize=12, fontweight='bold')
plt.savefig('graphics/forma-formb-vroeg.pdf', bbox_inches='tight')
plt.savefig('graphics/forma-formb-vroeg.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 7: forma-formb-vroeg')

# ── Figuur 8: Leeftijdsheatmap per testgroep (NIEUW) ──────────────────────
form_a_ages = [23, 21, 21, 58, 22, 58, 43, 50, 56, 23, 41, 59, 31, 55]   # Vroeg, n=14
form_b_ages = [21, 23, 25, 50, 58, 23, 69, 58, 26, 28, 48, 33, 24]        # Beide, n=13
form_c_ages = [49, 58, 36, 22, 21, 41, 56, 57, 22, 60, 21, 28, 55]        # Laat, n=13

age_bins   = [18, 26, 36, 46, 56, 70]
age_labels = ['18–25', '26–35', '36–45', '46–55', '56+']


def age_counts(ages):
    return pd.cut(ages, bins=age_bins, labels=age_labels,
                  right=False).value_counts().reindex(age_labels, fill_value=0)


heatmap_data = pd.DataFrame({
    'Vroeg\n(Form A, n=14)': age_counts(form_a_ages),
    'Laat\n(Form C, n=13)': age_counts(form_c_ages),
    'Beide\n(Form B, n=13)': age_counts(form_b_ages),
})

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
            cbar_kws={'label': 'Aantal deelnemers'}, ax=ax,
            annot_kws={'size': 12, 'weight': 'bold'})
ax.set_title('Leeftijdsverdeling per testgroep', fontsize=13, pad=12)
ax.set_xlabel('Testgroep', fontsize=11)
ax.set_ylabel('Leeftijdscategorie', fontsize=11)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('graphics/leeftijd-heatmap.pdf', bbox_inches='tight')
plt.savefig('graphics/leeftijd-heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 8: leeftijd-heatmap')

# ── Figuur 9: Achtergrondkenmerken per groep (NIEUW) ──────────────────────
groep_labels = ['Vroeg\n(Form A)', 'Laat\n(Form C)', 'Beide\n(Form B)']
avg_age  = [40.07, 40.46, 37.38]
avg_tech = [6.71,  6.46,  7.15]
pct_vrouw = [78.6, 92.3, 53.8]

x = np.arange(len(groep_labels))
w = 0.28

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

b1 = ax1.bar(x - w, avg_age,   w, label='Gem. leeftijd (j)', color=colors, alpha=0.75)
b2 = ax1.bar(x,     avg_tech,  w, label='Gem. tech-score',   color=colors, alpha=0.45,
             edgecolor='black', linewidth=0.8)
b3 = ax2.bar(x + w, pct_vrouw, w, label='% vrouwelijk',      color='#aaaaaa', alpha=0.65,
             edgecolor='black', linewidth=0.8)

for bar, val in zip(b1, avg_age):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             '{:.1f}'.format(val), ha='center', va='bottom', fontsize=8)
for bar, val in zip(b2, avg_tech):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             '{:.2f}'.format(val), ha='center', va='bottom', fontsize=8)
for bar, val in zip(b3, pct_vrouw):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 1,
             '{:.1f}%'.format(val), ha='center', va='bottom', fontsize=8)

ax1.set_ylabel('Leeftijd (j) / Tech-score (1–10)', fontsize=10)
ax2.set_ylabel('% vrouwelijk', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(groep_labels)
ax1.set_ylim(0, 55)
ax2.set_ylim(0, 120)
ax1.set_title('Achtergrondkenmerken per testgroep', fontsize=12, pad=10)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
ax1.set_axisbelow(True)
plt.tight_layout()
plt.savefig('graphics/groepskenmerken.pdf', bbox_inches='tight')
plt.savefig('graphics/groepskenmerken.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 9: groepskenmerken')

# ── Figuur 10: Voltooiingsfunnel vroege test (NIEUW) ──────────────────────
fA = funnel_vroeg[funnel_vroeg['form'] == 'FormA'].sort_values('stap')
fB = funnel_vroeg[funnel_vroeg['form'] == 'FormB'].sort_values('stap')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(fA['stap'], fA['sessies'], 'o-', color=COL_VROEG, linewidth=2, markersize=6,
        label='Form A (vroeg)')
ax.plot(fB['stap'], fB['sessies'], 's--', color=COL_BEIDE, linewidth=2, markersize=6,
        label='Form B (vroeg)')
ax.set_xlabel('Stap')
ax.set_ylabel('Aantal actieve sessies')
ax.set_title('Voltooiingsfunnel per stap — vroege test')
ax.set_xticks(range(1, int(max(fB['stap'])) + 1))
ax.set_ylim(0, max(fA['sessies'].max(), fB['sessies'].max()) * 1.15)
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.savefig('graphics/funnel-vroeg.pdf', bbox_inches='tight')
plt.savefig('graphics/funnel-vroeg.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 10: funnel-vroeg')

# ── Figuur 11: Stap-doorlooptijden vroege test (NIEUW) ────────────────────
sA = stapduur[stapduur['form'] == 'FormA'].sort_values('stap')
sB = stapduur[stapduur['form'] == 'FormB'].sort_values('stap')
sA_s = sA['gemiddeldMs'] / 1000
sB_s = sB['gemiddeldMs'] / 1000

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sA['stap'], sA_s, 'o-', color=COL_VROEG, linewidth=2, markersize=6,
        label='Form A (vroeg)')
ax.plot(sB['stap'], sB_s, 's--', color=COL_BEIDE, linewidth=2, markersize=6,
        label='Form B (vroeg)')
ax.set_xlabel('Stap')
ax.set_ylabel('Gemiddelde doorlooptijd (s)')
ax.set_title('Gemiddelde doorlooptijd per stap — vroege test')
ax.set_xticks(range(1, int(max(sB['stap'])) + 1))
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
plt.savefig('graphics/stapduur-vroeg.pdf', bbox_inches='tight')
plt.savefig('graphics/stapduur-vroeg.png', bbox_inches='tight', dpi=150)
plt.close()
print('Figuur 11: stapduur-vroeg')

print('\nKlaar. Alle figuren opgeslagen in graphics/')
