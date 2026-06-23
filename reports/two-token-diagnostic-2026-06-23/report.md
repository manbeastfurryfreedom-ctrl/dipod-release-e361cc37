# DiPOD Two-Token Diagnostic — Evidence Report

**Date:** 2026-06-23 · **Project:** DiPOD-release (`019ef269-13b2-7b21-84b2-fbf1b4b8a459`) · **Paper:** arXiv 2606.13795

## TL;DR

Minimal CPU reproduction of the DiPOD paper's controlled diagnostic (Section 4.1 / Appendix D). A fully-enumerable 2-token masked-diffusion policy is RL-post-trained with **exact** policy gradients (no Monte-Carlo). The paper's **double drift** is reproduced and measured directly: under FPO the ELBO–likelihood gap drifts up (0 → 0.24) and the proxy gradient misaligns from the true policy gradient (cosine 1.0 → 0.84); FPO+DiPOD keeps both controlled (gap 0.04, alignment 0.96) at no reward cost. Holds across 20 random rewards and a β-ablation over 12 seeds. **Verdict: REPRODUCED.**

## The Double Drift (conceptual)

<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg">
<text x="380" y="24" text-anchor="middle" font-size="14" font-weight="600" fill="#333">The Double Drift (DiPOD core claim)</text>
<rect x="30" y="60" width="280" height="90" rx="10" fill="#fde" stroke="#d62728" stroke-width="2"/>
<text x="170" y="85" text-anchor="middle" font-size="13" font-weight="600" fill="#222">1st Drift: ELBO drifts from log pi</text>
<text x="170" y="108" text-anchor="middle" font-size="12" fill="#444">ELBO = log pi - D_L</text>
<text x="170" y="128" text-anchor="middle" font-size="12" fill="#444">RL updates loosen D_L -&gt;</text>
<text x="170" y="143" text-anchor="middle" font-size="12" fill="#444">ELBO no longer tracks likelihood</text>
<path d="M 320 105 L 440 105" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<rect x="450" y="60" width="280" height="90" rx="10" fill="#fef" stroke="#9467bd" stroke-width="2"/>
<text x="590" y="85" text-anchor="middle" font-size="13" font-weight="600" fill="#222">2nd Drift: proxy grad drifts</text>
<text x="590" y="108" text-anchor="middle" font-size="12" fill="#444">grad ELBO = grad log pi - grad D_L</text>
<text x="590" y="128" text-anchor="middle" font-size="12" fill="#444">proxy PG misaligns from</text>
<text x="590" y="143" text-anchor="middle" font-size="12" fill="#444">true policy gradient</text>
<rect x="170" y="190" width="420" height="80" rx="10" fill="#e8f5e9" stroke="#2ca02c" stroke-width="2"/>
<text x="380" y="215" text-anchor="middle" font-size="13" font-weight="600" fill="#2e7d32">DiPOD: keep the bound tight on-policy</text>
<text x="380" y="238" text-anchor="middle" font-size="12" fill="#444">add beta * ELBO regularizer to each PG step</text>
<text x="380" y="258" text-anchor="middle" font-size="12" fill="#444">-&gt; D_L stays small -&gt; proxy grad stays aligned</text>
<path d="M 170 150 L 250 190" stroke="#2ca02c" stroke-width="2" marker-end="url(#ag)" stroke-dasharray="4,3"/>
<path d="M 590 150 L 510 190" stroke="#2ca02c" stroke-width="2" marker-end="url(#ag)" stroke-dasharray="4,3"/>
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#333"/></marker>
<marker id="ag" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#2ca02c"/></marker></defs>
</svg>

The paper's core claim: RL updates with an ELBO proxy can "cheat" by loosening the variational bound (D_L grows) rather than improving the true likelihood — so the proxy gradient drifts from the true policy gradient. DiPOD adds a `beta * ELBO` on-policy regularizer to keep the bound tight.

## Setup

- **Model:** 2-token masked discrete-diffusion policy, 6 logits `a..f` (paper Appendix D), init to 0 so ELBO = log π at start (tight, "pretrained").
- **Methods:** FPO (policy gradient on the ELBO score, paper Eq. 4) vs FPO+DiPOD (FPO + `beta * ELBO` regularizer, paper Algorithm 2 — the same term as `language/d1/diffu-grpo/diffu_grpo_trainer.py:160`).
- **Exactness:** the 4-sequence state space {AA,AB,BA,BB} is fully enumerable, so `log π`, `ELBO`, the gap `D^L = log π − ELBO`, and the **gradient alignment** `cos(∇ELBO-PG, ∇logπ-PG)` are all closed-form — no Monte-Carlo (as the paper specifies).
- **Hyperparameters (paper Appendix D):** lr=0.1, β_DiPOD=0.2, 1500 steps; reward r(AA)=0.8, r(AB)=1, r(BA)=0.7, r(BB)=1.

## Axis 1 — Paper single run (Figure 2 reproduction)

<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<text x="380.0" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#333">1st drift: discrepancy on AA grows under FPO</text>
<line x1="56" y1="364" x2="704" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56" x2="56" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="52" y1="364.0" x2="56" y2="364.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="364.0" x2="704" y2="364.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="368.0" text-anchor="end" font-size="11" fill="#666">-0.032</text>
<line x1="52" y1="287.0" x2="56" y2="287.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="287.0" x2="704" y2="287.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="291.0" text-anchor="end" font-size="11" fill="#666">0.047</text>
<line x1="52" y1="210.0" x2="56" y2="210.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="210.0" x2="704" y2="210.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="214.0" text-anchor="end" font-size="11" fill="#666">0.125</text>
<line x1="52" y1="132.99999999999997" x2="56" y2="132.99999999999997" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="132.99999999999997" x2="704" y2="132.99999999999997" stroke="#eee" stroke-width="1"/>
<text x="48" y="136.99999999999997" text-anchor="end" font-size="11" fill="#666">0.203</text>
<line x1="52" y1="56.0" x2="56" y2="56.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56.0" x2="704" y2="56.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="60.0" text-anchor="end" font-size="11" fill="#666">0.282</text>
<line x1="56.0" y1="364" x2="56.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="56.0" y="382" text-anchor="middle" font-size="11" fill="#666">0</text>
<line x1="218.0" y1="364" x2="218.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="218.0" y="382" text-anchor="middle" font-size="11" fill="#666">375</text>
<line x1="380.0" y1="364" x2="380.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="380.0" y="382" text-anchor="middle" font-size="11" fill="#666">750</text>
<line x1="542.0" y1="364" x2="542.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="542.0" y="382" text-anchor="middle" font-size="11" fill="#666">1125</text>
<line x1="704.0" y1="364" x2="704.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="704.0" y="382" text-anchor="middle" font-size="11" fill="#666">1500</text>
<text x="380.0" y="412" text-anchor="middle" font-size="12" fill="#444">training step</text>
<text x="14" y="210.0" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 14 210.0)">gap on AA (log pi - ELBO)</text>
<polyline points="56.0,332.9 66.8,332.9 77.6,332.6 88.4,332.2 99.2,331.7 110.0,330.9 120.8,329.9 131.6,328.8 142.4,327.3 153.2,325.7 164.0,323.9 174.8,321.8 185.6,319.5 196.4,316.9 207.2,314.2 218.0,311.2 228.8,308.1 239.6,304.7 250.4,301.2 261.2,297.5 272.0,293.6 282.8,289.6 293.6,285.5 304.4,281.2 315.2,276.9 326.0,272.4 336.8,267.8 347.6,263.1 358.4,258.4 369.2,253.6 380.0,248.7 390.8,243.8 401.6,238.8 412.4,233.8 423.2,228.8 434.0,223.8 444.8,218.7 455.6,213.6 466.4,208.5 477.2,203.4 488.0,198.3 498.8,193.2 509.6,188.1 520.4,183.0 531.2,177.9 542.0,172.8 552.8,167.8 563.6,162.7 574.4,157.7 585.2,152.7 596.0,147.7 606.8,142.8 617.6,137.8 628.4,132.9 639.2,128.1 650.0,123.2 660.8,118.4 671.6,113.6 682.4,108.8 693.2,104.1 704.0,99.4" fill="none" stroke="#d62728" stroke-width="2.5"/>
<circle cx="604" cy="76" r="4" fill="#d62728"/>
<text x="614" y="80" font-size="12" fill="#333">FPO (beta=0)</text>
<polyline points="56.0,332.9 66.8,332.9 77.6,332.7 88.4,332.3 99.2,331.9 110.0,331.3 120.8,330.7 131.6,329.9 142.4,329.0 153.2,328.1 164.0,327.1 174.8,326.1 185.6,325.0 196.4,323.9 207.2,322.7 218.0,321.6 228.8,320.4 239.6,319.2 250.4,318.0 261.2,316.9 272.0,315.7 282.8,314.6 293.6,313.4 304.4,312.3 315.2,311.3 326.0,310.2 336.8,309.2 347.6,308.2 358.4,307.2 369.2,306.3 380.0,305.4 390.8,304.5 401.6,303.6 412.4,302.8 423.2,302.0 434.0,301.3 444.8,300.5 455.6,299.8 466.4,299.2 477.2,298.5 488.0,297.9 498.8,297.3 509.6,296.7 520.4,296.1 531.2,295.6 542.0,295.1 552.8,294.6 563.6,294.1 574.4,293.7 585.2,293.2 596.0,292.8 606.8,292.4 617.6,292.0 628.4,291.6 639.2,291.3 650.0,290.9 660.8,290.6 671.6,290.3 682.4,290.0 693.2,289.7 704.0,289.4" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
<circle cx="604" cy="96" r="4" fill="#1f77b4"/>
<text x="614" y="100" font-size="12" fill="#333">FPO+DiPOD (beta=0.2)</text>
</svg>

*The 1st drift: under FPO the ELBO–likelihood discrepancy on AA climbs to 0.2375; DiPOD holds it at 0.0443 (5.4× smaller). Matches the paper's Figure 2.*

<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<text x="380.0" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#333">2nd drift: gradient alignment falls under FPO</text>
<line x1="56" y1="364" x2="704" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56" x2="56" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="52" y1="364.0" x2="56" y2="364.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="364.0" x2="704" y2="364.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="368.0" text-anchor="end" font-size="11" fill="#666">0.805</text>
<line x1="52" y1="287.0" x2="56" y2="287.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="287.0" x2="704" y2="287.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="291.0" text-anchor="end" font-size="11" fill="#666">0.859</text>
<line x1="52" y1="210.0" x2="56" y2="210.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="210.0" x2="704" y2="210.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="214.0" text-anchor="end" font-size="11" fill="#666">0.912</text>
<line x1="52" y1="133.0" x2="56" y2="133.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="133.0" x2="704" y2="133.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="137.0" text-anchor="end" font-size="11" fill="#666">0.966</text>
<line x1="52" y1="56.0" x2="56" y2="56.0" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56.0" x2="704" y2="56.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="60.0" text-anchor="end" font-size="11" fill="#666">1.020</text>
<line x1="56.0" y1="364" x2="56.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="56.0" y="382" text-anchor="middle" font-size="11" fill="#666">0</text>
<line x1="218.0" y1="364" x2="218.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="218.0" y="382" text-anchor="middle" font-size="11" fill="#666">375</text>
<line x1="380.0" y1="364" x2="380.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="380.0" y="382" text-anchor="middle" font-size="11" fill="#666">750</text>
<line x1="542.0" y1="364" x2="542.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="542.0" y="382" text-anchor="middle" font-size="11" fill="#666">1125</text>
<line x1="704.0" y1="364" x2="704.0" y2="368" stroke="#555" stroke-width="1.5"/>
<text x="704.0" y="382" text-anchor="middle" font-size="11" fill="#666">1500</text>
<text x="380.0" y="412" text-anchor="middle" font-size="12" fill="#444">training step</text>
<text x="14" y="210.0" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 14 210.0)">cos(proxy PG, true PG)</text>
<polyline points="56.0,84.4 66.8,84.4 77.6,84.4 88.4,84.5 99.2,84.5 110.0,84.6 120.8,84.7 131.6,84.9 142.4,85.2 153.2,85.7 164.0,86.2 174.8,87.0 185.6,87.9 196.4,89.1 207.2,90.4 218.0,92.0 228.8,93.9 239.6,96.0 250.4,98.4 261.2,101.0 272.0,103.9 282.8,107.1 293.6,110.5 304.4,114.2 315.2,118.2 326.0,122.3 336.8,126.7 347.6,131.3 358.4,136.0 369.2,141.0 380.0,146.1 390.8,151.3 401.6,156.7 412.4,162.1 423.2,167.7 434.0,173.3 444.8,179.0 455.6,184.8 466.4,190.6 477.2,196.4 488.0,202.2 498.8,208.1 509.6,213.9 520.4,219.7 531.2,225.5 542.0,231.3 552.8,237.0 563.6,242.7 574.4,248.4 585.2,254.0 596.0,259.5 606.8,265.0 617.6,270.4 628.4,275.8 639.2,281.0 650.0,286.3 660.8,291.4 671.6,296.5 682.4,301.5 693.2,306.4 704.0,311.2" fill="none" stroke="#d62728" stroke-width="2.5"/>
<circle cx="604" cy="76" r="4" fill="#d62728"/>
<text x="614" y="80" font-size="12" fill="#333">FPO (beta=0)</text>
<polyline points="56.0,84.4 66.8,84.4 77.6,84.4 88.4,84.4 99.2,84.5 110.0,84.5 120.8,84.6 131.6,84.8 142.4,85.0 153.2,85.2 164.0,85.6 174.8,85.9 185.6,86.4 196.4,87.0 207.2,87.6 218.0,88.3 228.8,89.1 239.6,89.9 250.4,90.8 261.2,91.8 272.0,92.8 282.8,93.9 293.6,95.0 304.4,96.2 315.2,97.4 326.0,98.6 336.8,99.9 347.6,101.1 358.4,102.4 369.2,103.7 380.0,105.1 390.8,106.4 401.6,107.7 412.4,109.0 423.2,110.4 434.0,111.7 444.8,113.0 455.6,114.3 466.4,115.6 477.2,116.9 488.0,118.1 498.8,119.4 509.6,120.6 520.4,121.8 531.2,123.0 542.0,124.2 552.8,125.4 563.6,126.5 574.4,127.6 585.2,128.7 596.0,129.8 606.8,130.9 617.6,132.0 628.4,133.0 639.2,134.0 650.0,135.0 660.8,136.0 671.6,137.0 682.4,137.9 693.2,138.8 704.0,139.7" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
<circle cx="604" cy="96" r="4" fill="#1f77b4"/>
<text x="614" y="100" font-size="12" fill="#333">FPO+DiPOD (beta=0.2)</text>
</svg>

*The 2nd drift, measured directly: cosine similarity between the FPO proxy gradient and the TRUE policy gradient falls to 0.842 under FPO; DiPOD keeps it at 0.961. This is the mechanistic heart of the paper — measurable here because `log π` is tractable, which it is not at LLaDA-8B scale.*

| Method | final gap(AA) | final mean gap | final cos(proxy,true) | final reward |
|---|---|---|---|---|
| FPO (β=0) | 0.2375 | 0.0291 | 0.8420 | 0.9733 |
| FPO+DiPOD (β=0.2) | 0.0443 | 0.0059 | 0.9615 | 0.9741 |

## Axis 2 — Reward robustness (20 random rewards)

<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<text x="380.0" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#333">DiPOD controls the gap below FPO across random rewards</text>
<line x1="56" y1="364" x2="704" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56" x2="56" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="56.0" y1="364.0" x2="704.0" y2="56.0" stroke="#999" stroke-dasharray="5,4" stroke-width="1.5"/>
<circle cx="74.87416621250577" cy="355.0289438417571" r="5" fill="#d62728" opacity="0.75"/>
<circle cx="134.66044396387167" cy="351.099249982185" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="161.2905102039041" cy="329.35475163049745" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="74.87484595402535" cy="355.02395554271425" r="5" fill="#d62728" opacity="0.75"/>
<circle cx="74.87500822159389" cy="355.0282990383505" r="5" fill="#d62728" opacity="0.75"/>
<circle cx="646.8067078552516" cy="239.51440241955237" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="516.0236879311628" cy="291.5672605685562" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="74.87732560868562" cy="355.01521274754737" r="5" fill="#d62728" opacity="0.75"/>
<circle cx="425.91206077242515" cy="281.7335324345752" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="122.85598538965085" cy="343.6881707430184" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="81.15169597134964" cy="354.48862496009343" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="442.9682777799324" cy="292.48402531273103" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="174.9095103522447" cy="331.51444266915126" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="74.87397963246694" cy="355.02909595822155" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="190.15023944394045" cy="337.0108835301745" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="82.81467810329292" cy="354.7572219995256" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="148.26356406280945" cy="349.9419029425795" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="159.18853672692651" cy="349.9976960320454" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="366.0266211201291" cy="300.7049241457866" r="5" fill="#2ca02c" opacity="0.75"/>
<circle cx="112.8572623283312" cy="346.3408626471093" r="5" fill="#2ca02c" opacity="0.75"/>
<text x="56.0" y="382" text-anchor="middle" font-size="11" fill="#666">-0.02</text>
<text x="48" y="368.0" text-anchor="end" font-size="11" fill="#666">-0.02</text>
<line x1="56" y1="364.0" x2="704" y2="364.0" stroke="#eee" stroke-width="1"/>
<text x="218.0" y="382" text-anchor="middle" font-size="11" fill="#666">0.15</text>
<text x="48" y="291.0" text-anchor="end" font-size="11" fill="#666">0.15</text>
<line x1="56" y1="287.0" x2="704" y2="287.0" stroke="#eee" stroke-width="1"/>
<text x="380.0" y="382" text-anchor="middle" font-size="11" fill="#666">0.33</text>
<text x="48" y="214.0" text-anchor="end" font-size="11" fill="#666">0.33</text>
<line x1="56" y1="210.0" x2="704" y2="210.0" stroke="#eee" stroke-width="1"/>
<text x="542.0" y="382" text-anchor="middle" font-size="11" fill="#666">0.50</text>
<text x="48" y="137.0" text-anchor="end" font-size="11" fill="#666">0.50</text>
<line x1="56" y1="133.0" x2="704" y2="133.0" stroke="#eee" stroke-width="1"/>
<text x="704.0" y="382" text-anchor="middle" font-size="11" fill="#666">0.67</text>
<text x="48" y="60.0" text-anchor="end" font-size="11" fill="#666">0.67</text>
<line x1="56" y1="56.0" x2="704" y2="56.0" stroke="#eee" stroke-width="1"/>
<text x="380.0" y="412" text-anchor="middle" font-size="12" fill="#444">FPO final gap(AA)</text>
<text x="14" y="210.0" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 14 210.0)">DiPOD final gap(AA)</text>
</svg>

*Each point is one random reward table (uniform draws over the 4 sequences). Green = DiPOD controls the gap below FPO for that reward.*

DiPOD gap < FPO gap in **16/20** (80%) of random rewards; DiPOD alignment > FPO alignment in **20/20** (100%). The drift is generic, not an artefact of the paper's single hand-chosen reward.

## Axis 3 — β-ablation (paper Appendix E.2, 12 seeds each)

<div style="display:flex;gap:12px;flex-wrap:wrap">
<div style="flex:1;min-width:360px">
<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<text x="380.0" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#333">beta vs gap(AA) [mean +/- std]</text>
<line x1="56" y1="364" x2="704" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56" x2="56" y2="364" stroke="#555" stroke-width="1.5"/>
<text x="48" y="368.0" text-anchor="end" font-size="11" fill="#666">0.000</text>
<line x1="56" y1="364.0" x2="704" y2="364.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="291.0" text-anchor="end" font-size="11" fill="#666">0.112</text>
<line x1="56" y1="287.0" x2="704" y2="287.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="214.0" text-anchor="end" font-size="11" fill="#666">0.225</text>
<line x1="56" y1="210.0" x2="704" y2="210.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="137.0" text-anchor="end" font-size="11" fill="#666">0.337</text>
<line x1="56" y1="133.0" x2="704" y2="133.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="60.0" text-anchor="end" font-size="11" fill="#666">0.449</text>
<line x1="56" y1="56.0" x2="704" y2="56.0" stroke="#eee" stroke-width="1"/>
<text x="56.0" y="382" text-anchor="middle" font-size="11" fill="#666">0</text>
<text x="114.9090909090909" y="382" text-anchor="middle" font-size="11" fill="#666">0.05</text>
<text x="173.8181818181818" y="382" text-anchor="middle" font-size="11" fill="#666">0.1</text>
<text x="291.6363636363636" y="382" text-anchor="middle" font-size="11" fill="#666">0.2</text>
<text x="645.0909090909091" y="382" text-anchor="middle" font-size="11" fill="#666">0.5</text>
<line x1="56.0" y1="437.7421235766572" x2="56.0" y2="84.0" stroke="#d62728" stroke-width="2"/>
<line x1="51.0" y1="437.7421235766572" x2="61.0" y2="437.7421235766572" stroke="#d62728" stroke-width="2"/>
<line x1="51.0" y1="84.0" x2="61.0" y2="84.0" stroke="#d62728" stroke-width="2"/>
<circle cx="56.0" cy="260.87106178832863" r="5" fill="#d62728"/>
<line x1="114.9090909090909" y1="431.1366827000055" x2="114.9090909090909" y2="147.32472686794142" stroke="#d62728" stroke-width="2"/>
<line x1="109.9090909090909" y1="431.1366827000055" x2="119.9090909090909" y2="431.1366827000055" stroke="#d62728" stroke-width="2"/>
<line x1="109.9090909090909" y1="147.32472686794142" x2="119.9090909090909" y2="147.32472686794142" stroke="#d62728" stroke-width="2"/>
<circle cx="114.9090909090909" cy="289.23070478397347" r="5" fill="#d62728"/>
<line x1="173.8181818181818" y1="421.36950240783005" x2="173.8181818181818" y2="193.75207460609053" stroke="#d62728" stroke-width="2"/>
<line x1="168.8181818181818" y1="421.36950240783005" x2="178.8181818181818" y2="421.36950240783005" stroke="#d62728" stroke-width="2"/>
<line x1="168.8181818181818" y1="193.75207460609053" x2="178.8181818181818" y2="193.75207460609053" stroke="#d62728" stroke-width="2"/>
<circle cx="173.8181818181818" cy="307.56078850696025" r="5" fill="#d62728"/>
<line x1="291.6363636363636" y1="403.5465106289712" x2="291.6363636363636" y2="255.11678930824846" stroke="#d62728" stroke-width="2"/>
<line x1="286.6363636363636" y1="403.5465106289712" x2="296.6363636363636" y2="403.5465106289712" stroke="#d62728" stroke-width="2"/>
<line x1="286.6363636363636" y1="255.11678930824846" x2="296.6363636363636" y2="255.11678930824846" stroke="#d62728" stroke-width="2"/>
<circle cx="291.6363636363636" cy="329.33164996860984" r="5" fill="#d62728"/>
<line x1="645.0909090909091" y1="377.16871140382176" x2="645.0909090909091" y2="327.49470435322206" stroke="#d62728" stroke-width="2"/>
<line x1="640.0909090909091" y1="377.16871140382176" x2="650.0909090909091" y2="377.16871140382176" stroke="#d62728" stroke-width="2"/>
<line x1="640.0909090909091" y1="327.49470435322206" x2="650.0909090909091" y2="327.49470435322206" stroke="#d62728" stroke-width="2"/>
<circle cx="645.0909090909091" cy="352.3317078785219" r="5" fill="#d62728"/>
<text x="380.0" y="412" text-anchor="middle" font-size="12" fill="#444">beta (DiPOD)</text>
<text x="14" y="210.0" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 14 210.0)">final gap(AA)</text>
</svg>
</div>
<div style="flex:1;min-width:360px">
<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg">
<text x="380.0" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#333">beta vs alignment [mean +/- std]</text>
<line x1="56" y1="364" x2="704" y2="364" stroke="#555" stroke-width="1.5"/>
<line x1="56" y1="56" x2="56" y2="364" stroke="#555" stroke-width="1.5"/>
<text x="48" y="368.0" text-anchor="end" font-size="11" fill="#666">0.000</text>
<line x1="56" y1="364.0" x2="704" y2="364.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="291.0" text-anchor="end" font-size="11" fill="#666">0.273</text>
<line x1="56" y1="287.0" x2="704" y2="287.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="214.0" text-anchor="end" font-size="11" fill="#666">0.546</text>
<line x1="56" y1="210.0" x2="704" y2="210.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="137.0" text-anchor="end" font-size="11" fill="#666">0.819</text>
<line x1="56" y1="133.0" x2="704" y2="133.0" stroke="#eee" stroke-width="1"/>
<text x="48" y="60.0" text-anchor="end" font-size="11" fill="#666">1.093</text>
<line x1="56" y1="56.0" x2="704" y2="56.0" stroke="#eee" stroke-width="1"/>
<text x="56.0" y="382" text-anchor="middle" font-size="11" fill="#666">0</text>
<text x="114.9090909090909" y="382" text-anchor="middle" font-size="11" fill="#666">0.05</text>
<text x="173.8181818181818" y="382" text-anchor="middle" font-size="11" fill="#666">0.1</text>
<text x="291.6363636363636" y="382" text-anchor="middle" font-size="11" fill="#666">0.2</text>
<text x="645.0909090909091" y="382" text-anchor="middle" font-size="11" fill="#666">0.5</text>
<line x1="56.0" y1="158.44801761241018" x2="56.0" y2="109.34177746960543" stroke="#1f77b4" stroke-width="2"/>
<line x1="51.0" y1="158.44801761241018" x2="61.0" y2="158.44801761241018" stroke="#1f77b4" stroke-width="2"/>
<line x1="51.0" y1="109.34177746960543" x2="61.0" y2="109.34177746960543" stroke="#1f77b4" stroke-width="2"/>
<circle cx="56.0" cy="133.8948975410078" r="5" fill="#1f77b4"/>
<line x1="114.9090909090909" y1="147.89053831139918" x2="114.9090909090909" y2="101.47110964513098" stroke="#1f77b4" stroke-width="2"/>
<line x1="109.9090909090909" y1="147.89053831139918" x2="119.9090909090909" y2="147.89053831139918" stroke="#1f77b4" stroke-width="2"/>
<line x1="109.9090909090909" y1="101.47110964513098" x2="119.9090909090909" y2="101.47110964513098" stroke="#1f77b4" stroke-width="2"/>
<circle cx="114.9090909090909" cy="124.6808239782651" r="5" fill="#1f77b4"/>
<line x1="173.8181818181818" y1="138.05714041053994" x2="173.8181818181818" y2="96.03806004375781" stroke="#1f77b4" stroke-width="2"/>
<line x1="168.8181818181818" y1="138.05714041053994" x2="178.8181818181818" y2="138.05714041053994" stroke="#1f77b4" stroke-width="2"/>
<line x1="168.8181818181818" y1="96.03806004375781" x2="178.8181818181818" y2="96.03806004375781" stroke="#1f77b4" stroke-width="2"/>
<circle cx="173.8181818181818" cy="117.04760022714888" r="5" fill="#1f77b4"/>
<line x1="291.6363636363636" y1="121.48635927106622" x2="291.6363636363636" y2="89.73237478354184" stroke="#1f77b4" stroke-width="2"/>
<line x1="286.6363636363636" y1="121.48635927106622" x2="296.6363636363636" y2="121.48635927106622" stroke="#1f77b4" stroke-width="2"/>
<line x1="286.6363636363636" y1="89.73237478354184" x2="296.6363636363636" y2="89.73237478354184" stroke="#1f77b4" stroke-width="2"/>
<circle cx="291.6363636363636" cy="105.60936702730402" r="5" fill="#1f77b4"/>
<line x1="645.0909090909091" y1="96.30164212572788" x2="645.0909090909091" y2="84.0" stroke="#1f77b4" stroke-width="2"/>
<line x1="640.0909090909091" y1="96.30164212572788" x2="650.0909090909091" y2="96.30164212572788" stroke="#1f77b4" stroke-width="2"/>
<line x1="640.0909090909091" y1="84.0" x2="650.0909090909091" y2="84.0" stroke="#1f77b4" stroke-width="2"/>
<circle cx="645.0909090909091" cy="90.150821062864" r="5" fill="#1f77b4"/>
<text x="380.0" y="412" text-anchor="middle" font-size="12" fill="#444">beta (DiPOD)</text>
<text x="14" y="210.0" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 14 210.0)">final cos(proxy,true)</text>
</svg>
</div>
</div>

*Increasing β monotonically reduces the gap and raises alignment, with reward roughly flat — consistent with the paper's Appendix E.2 (a moderate β controls drift without hurting reward).*

| β | n | final gap(AA) | final cos(proxy,true) | final reward |
|---|---|---|---|---|
| 0 | 12 | 0.1505±0.2581 | 0.8162±0.0871 | 0.7383±0.1339 |
| 0.05 | 12 | 0.1091±0.2071 | 0.8489±0.0823 | 0.7388±0.1339 |
| 0.1 | 12 | 0.0823±0.1661 | 0.8760±0.0745 | 0.7390±0.1340 |
| 0.2 | 12 | 0.0506±0.1083 | 0.9166±0.0563 | 0.7393±0.1340 |
| 0.5 | 12 | 0.0170±0.0362 | 0.9714±0.0218 | 0.7391±0.1342 |

## Verdict

**REPRODUCED.** Single-run: DiPOD gap 0.0443 < FPO 0.2375, alignment 0.961 > FPO 0.842; robustness: DiPOD controls gap in 80% and alignment in 100% of 20 random rewards; ablation: monotonic gap reduction and alignment recovery with β, reward flat.

## Scope & honesty

- **FPO side only.** SPG/EUBO deliberately excluded: the SPG curve needs a tractable EUBO upper bound; a correct one for this two-token model requires more care than is warranted here, and an unaudited EUBO would risk a misleading curve.
- This is the **minimal** illustration of the paper's central mechanism (the two-token diagnostic). The full language (LLaDA-8B on GSM8K/MATH500/Countdown/Sudoku) and control (IsaacLab G1 motion tracking) experiments live under `language/` and `fpo-control/` and need 8×H100 / Isaac Sim respectively.

## Reproduce

```bash
bash twotoken/run.sh
```

Writes `EVAL.md` at the repo root plus `history.json` / `paper_single.json` / `reward_robust.json` / `beta_ablation.json` / `twotoken_diagnostic.png` under `.openresearch/artifacts/`. Run: `019ef2a4-5220-7ad8-b405-114211c145af` (status: done).
