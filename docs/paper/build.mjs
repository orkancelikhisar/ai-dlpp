/**
 * Renders the paper and the executive summary to PDF with Playwright's Chromium,
 * and screenshots the figure contact sheet for visual checking.
 *   node docs/paper/build.mjs            -> paper.pdf, executive-summary.pdf, _contact.png
 */
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";
const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "../../apps/eval/package.json"));
const { chromium } = require("@playwright/test");

const jobs = [
  { html: "paper.html", pdf: "paper.pdf" },
  { html: "executive-summary.html", pdf: "executive-summary.pdf" },
];
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  for (const j of jobs) {
    const src = join(here, j.html);
    if (!existsSync(src)) { console.log(`skip ${j.html} (missing)`); continue; }
    await page.goto(pathToFileURL(src).href, { waitUntil: "networkidle" });
    await page.emulateMedia({ media: "print" });
    const footer = j.pdf === "paper.pdf"
      ? { displayHeaderFooter: true, headerTemplate: "<span></span>",
          footerTemplate: "<div style='width:100%;font:7pt Helvetica,Arial,sans-serif;color:#666;padding:0 13mm;display:flex;justify-content:space-between'><span>AI-DLPP · capability-ceiling study · draft v2</span><span><span class='pageNumber'></span> / <span class='totalPages'></span></span></div>",
          margin: { top: "14mm", bottom: "16mm", left: "13mm", right: "13mm" } }
      : {};
    await page.pdf({ path: join(here, j.pdf), format: "A4", printBackground: true, preferCSSPageSize: !footer.displayHeaderFooter, ...footer });
    console.log(`wrote ${j.pdf}`);
  }
  const contact = join(here, "figures/_contact.html");
  if (existsSync(contact)) {
    await page.setViewportSize({ width: 760, height: 900 });
    await page.goto(pathToFileURL(contact).href, { waitUntil: "networkidle" });
    await page.screenshot({ path: join(here, "figures/_contact.png"), fullPage: true });
    console.log("wrote figures/_contact.png");
  }
} finally { await browser.close(); }
