ROULETTE AI ANALYZER — Updated app with TensorFlow.js AI plugin

FILES:
- index.html       App UI, results, stats and CSV/JSON backup tools
- ai-plugin.js     TensorFlow.js training / prediction module
- manifest.webmanifest, sw.js, icon-192.png, icon-512.png    Installable web app files

HOW TO UPDATE YOUR EXISTING SITE:
1. Open your OLD app first and click Backup JSON if already available. If it only has Export CSV, export that CSV; do not reset old data. Use your browser's existing site address.
2. Upload ALL six app files to the exact SAME website folder/origin as your old index.html, replacing the old files. If your current website already has another custom file, keep it separately.
3. Reload the site. Existing rai_results, rai_attempts and rai_current_picks are kept in localStorage on the SAME browser + origin.
4. Tap Train AI model to download TensorFlow.js the first time. This requires an internet connection.
5. Wait for training to complete. AI picks and frequency picks show separately. Use the menu to choose which method is recorded in the main Top-5 hit-rate.
6. AI model may be saved to this browser's IndexedDB. If storage is disabled, train again after refresh.
7. On iPhone Safari: Share > Add to Home Screen (HTTPS website recommended).

IMPORTANT:
- If you open the upgraded files under a NEW site/URL, browser localStorage does NOT move automatically. A JSON backup from this upgraded version can be imported into another install, but your old version's CSV only has the results, not the historical prediction records.
- You can run the interface offline after first hosting, but training the neural network requires the TensorFlow.js library to load. This build uses a public CDN and does not upload your roulette data.
- AI training is allowed from 50 outcomes as a DEMO. It is far too little to establish a meaningful pattern. Fair RNG outcomes are independent; no AI can guarantee a future result.
- The five-spin metric checks complete windows only; overlapping windows are not independent.

No casino server access or game exploit is present.

MIGRATING FROM THE OLD VERSION TO A DIFFERENT SITE:
Use the old app's Export CSV button. In the upgraded app click Import CSV.
This restores winning results only, because the original CSV omits the old prediction/accuracy records.
After upgrading, use Backup JSON to preserve prediction records too.
