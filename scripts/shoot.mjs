import { chromium } from 'playwright'
// Usage: node scripts/shoot.mjs <url> <out.png>
// Needs Playwright: npm i -D playwright && npx playwright install chromium
const [,, url, out] = process.argv
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errs = []
page.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 200)) })
page.on('pageerror', e => errs.push('pageerror: ' + e.message.slice(0, 200)))
await page.goto(url, { waitUntil: 'networkidle' })
await page.waitForTimeout(2500)
await page.screenshot({ path: out })
console.log(out, errs.length ? 'ERRORS: ' + errs.join(' | ') : 'no console errors')
await browser.close()
