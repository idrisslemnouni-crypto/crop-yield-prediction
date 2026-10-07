"""Self-contained browser report of observed identity counts, without target estimates."""

from html import escape


def render_html(evidence, crosswalk):
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
        for values in crosswalk[
            ["COUNTY_ID", "census_county_name", "state_ansi", "county_ansi", "fips", "status"]
        ].itertuples(index=False, name=None)
    )
    maximum = max(row["training_counties"] for row in evidence["by_state"])
    bars = "".join(
        f'<div class="bar-row"><strong>{escape(row["state"])}</strong><div class="track"><div class="bar" style="width:{100 * row["training_counties"] / maximum:.2f}%"></div></div><span>{row["matched_counties"]} / {row["training_counties"]}</span></div>'
        for row in evidence["by_state"]
    )
    return f'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Identités de comtés · Audit rendement agricole</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#19372d;background:#f4f6f1}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1100px;margin:40px auto;padding:0 24px}}h1{{font-size:clamp(30px,5vw,48px);line-height:1.15;max-width:820px}}p{{line-height:1.6}}.tag{{letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#51705f}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}section,.card{{background:white;border:1px solid #d8e2d7;border-radius:10px;padding:24px;margin:20px 0}}.card{{margin:0}}.number{{font-size:36px;font-weight:700}}.notice{{background:#fff4d9;border-color:#dfc58b}}.bar-row{{display:grid;grid-template-columns:40px 1fr 90px;align-items:center;gap:14px;margin:12px 0}}.track{{background:#edf1e9;height:18px;border-radius:4px}}.bar{{background:#287455;height:18px;border-radius:4px}}input{{padding:12px;width:min(100%,500px);font:inherit;border:1px solid #94af9d;border-radius:5px;margin:8px 0 16px}}.table-wrap{{overflow:auto;max-height:520px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e1e7df;white-space:nowrap}}th{{position:sticky;top:0;background:#eaf0e6}}code{{overflow-wrap:anywhere}}a{{color:#176345}}footer{{font-size:13px;line-height:1.6;margin:26px 0}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}main{{padding:0 14px}}section{{padding:18px}}}}
</style></head><body><main><div class="tag">Data Science × Agriculture · 7 octobre 2026</div>
<h1>Relier les noms historiques aux codes officiels.</h1>
<p>Audit des identifiants de l'entraînement 2000–2015. Les correspondances utilisent le nom du comté dans son État, après normalisation documentée, et conservent les zéros des codes géographiques.</p>
<div class="cards"><div class="card"><div class="number">{evidence["training_counties"]}</div>comtés historiques</div><div class="card"><div class="number">{evidence["matched_counties"]}</div>correspondances uniques dans la référence</div><div class="card"><div class="number">{evidence["training_rows"]:,}</div>observations historiques d'entraînement</div></div>
<section class="notice"><h2>Couverture USDA non vérifiée</h2><p>La consultation des comptes USDA nécessite une clé API. Aucun rendement réservé 2024–2025 ni nouveau score n'est obtenu. La correspondance Census ne valide pas encore l'identité NASS, la stabilité des frontières ou les futurs prédicteurs.</p></section>
<section><h2>Correspondances par État</h2><p>Comtés associés / comtés historiques. Il s'agit du sous-ensemble historique, pas de la couverture agricole complète de chaque État.</p>{bars}</section>
<section><h2>Examiner les identifiants</h2><label for="county-search">Rechercher un comté ou un code</label><br><input id="county-search" type="search" placeholder="Exemple : IA_ADAIR ou 19001"><p id="visible-count" role="status">{len(crosswalk)} lignes affichées</p><div class="table-wrap"><table><thead><tr><th>Identifiant historique</th><th>Nom Census</th><th>État ANSI</th><th>Comté ANSI</th><th>FIPS</th><th>État du rapprochement</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>Provenance et limites</h2><p><a href="{escape(evidence["reference_url"], quote=True)}">U.S. Census Bureau, référence des codes de comtés</a>, téléchargée le 7 octobre 2026 et figée par checksum. Le dossier codes2020 ne prouve pas une frontière inchangée jusqu'en 2025.</p><p>SHA-256 : <code>{escape(evidence["reference_sha256"])}</code></p><p>Historique : Paudel, de Wit et Boogaard, <a href="https://doi.org/10.5281/zenodo.7751191">Zenodo 7751191</a>, CC BY 4.0. Les résultats prédictifs antérieurs restent conservés. {escape(evidence["scope"])}</p></section>
<footer>Développement assisté par IA · Python et notebook Jupyter · Contrôle R séparé dans la CI · Aucune validation terrain ou maîtrise personnelle attestée.</footer>
</main><script>
const input=document.getElementById('county-search'),rows=[...document.querySelectorAll('tbody tr')];input.addEventListener('input',()=>{{const query=input.value.toUpperCase().trim();let count=0;for(const row of rows){{row.hidden=!row.textContent.toUpperCase().includes(query);if(!row.hidden)count++;}}document.getElementById('visible-count').textContent=count+' lignes affichées';}});
</script></body></html>'''
