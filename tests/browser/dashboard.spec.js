const { test, expect } = require('@playwright/test');
const report = require('../../dashboard/data/runs.json');

test('report renders, filters, sorts, searches, and persists its theme', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#content')).toBeVisible();
  await expect(page.locator('#results-table tbody tr')).toHaveCount(16);
  await expect(page.locator('#plot-metrics svg')).toBeVisible();
  const metric = page.getByRole('button', { name: 'Token F1', exact: true });
  await metric.click();
  await expect(metric).toHaveAttribute('aria-pressed', 'false');
  await page.locator('#checkpoint-search').fill('1200');
  await expect(page.locator('#results-table tbody tr')).toHaveCount(1);
  await expect(page.locator('#results-table tbody')).toContainText('best');
  await page.locator('#checkpoint-search').fill('nothing matches');
  await expect(page.locator('#table-status')).toContainText('No matching');
  await page.locator('#checkpoint-search').fill('');
  await page.getByRole('button', { name: 'Step ↑', exact: true }).click();
  await expect(page.locator('#results-table tbody tr').first()).toContainText('1500');
  await page.locator('#theme-toggle').click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  expect(errors).toEqual([]);
});

test('exports data and safely imports a new report', async ({ page }) => {
  await page.goto('/');
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#export-report').click();
  expect((await downloadEvent).suggestedFilename()).toBe('llmft-checkpoints.csv');
  const imported = structuredClone(report);
  imported.demo = false;
  imported.run.name = 'My local experiment';
  imported.summary.best.name = '<img src=x onerror="window.injected=true">';
  imported.checkpoints[12].name = imported.summary.best.name;
  await page.locator('#report-file').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(imported)) });
  await expect(page.locator('#run-title')).toHaveText('My local experiment');
  await expect(page.locator('#report-badge')).toHaveText('Evaluation report');
  await expect(page.locator('.tile').nth(1)).toContainText('<img');
  expect(await page.evaluate(() => window.injected)).toBeUndefined();
  await page.locator('#report-file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"checkpoints":[null]}') });
  await expect(page.locator('#notification')).toContainText('Could not import');
  await expect(page.locator('#run-title')).toHaveText('My local experiment');
});

test('selects the best sample, not the first training checkpoint', async ({ page }) => {
  const data = structuredClone(report);
  data.checkpoints[1].samples = [{ prompt: 'different', reference: 'x', prediction: 'wrong comparison' }];
  await page.route('**/data/runs.json', route => route.fulfill({ json: data }));
  await page.goto('/');
  await expect(page.locator('#samples')).toContainText('checkpoint-1200');
  await expect(page.locator('#samples')).not.toContainText('wrong comparison');
});

test('shows a useful load failure and handles a single checkpoint', async ({ page }) => {
  await page.route('**/data/runs.json', route => route.fulfill({ status: 500, body: 'failure' }));
  await page.goto('/');
  await expect(page.locator('#state')).toContainText('Could not load');
  await page.unroute('**/data/runs.json');
  const data = structuredClone(report);
  data.checkpoints = [data.checkpoints[0]];
  data.summary = {};
  await page.route('**/data/runs.json', route => route.fulfill({ json: data }));
  await page.reload();
  await expect(page.locator('#content')).toBeVisible();
  await expect(page.locator('#plot-loss')).toContainText('No loss recorded');
  expect(await page.locator('svg').evaluateAll(nodes => nodes.every(node => !/NaN|Infinity/.test(node.innerHTML)))).toBeTruthy();
});
