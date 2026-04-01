# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e.spec.js >> homepage has title and create-run form
- Location: tests/e2e.spec.js:3:1

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: getByRole('heading', { name: 'synthDataLab Production UI' })
Expected: visible
Timeout: 5000ms
Error: element(s) not found

Call log:
  - Expect "toBeVisible" with timeout 5000ms
  - waiting for getByRole('heading', { name: 'synthDataLab Production UI' })

```

# Page snapshot

```yaml
- generic [ref=e3]:
  - banner [ref=e4]:
    - heading "synthDataLab Production Dashboard" [level=1] [ref=e5]
    - paragraph [ref=e6]: Monitor and launch synthetic dataset generation runs with health checks, logs, and pipeline controls.
  - generic [ref=e7]:
    - generic [ref=e8]:
      - heading "Total runs" [level=3] [ref=e9]
      - strong [ref=e10]: "0"
    - generic [ref=e11]:
      - heading "Running" [level=3] [ref=e12]
      - strong [ref=e13]: "0"
    - generic [ref=e14]:
      - heading "Succeeded" [level=3] [ref=e15]
      - strong [ref=e16]: "0"
    - generic [ref=e17]:
      - heading "Failed" [level=3] [ref=e18]
      - strong [ref=e19]: "0"
    - generic [ref=e20]:
      - heading "Canceled" [level=3] [ref=e21]
      - strong [ref=e22]: "0"
  - generic [ref=e23]:
    - heading "What this dashboard does" [level=2] [ref=e24]
    - paragraph [ref=e25]: synthDataLab is a production-grade synthetic dataset pipeline manager. You can create runs, monitor live status, inspect logs, and cancel in-progress jobs.
    - list [ref=e26]:
      - listitem [ref=e27]:
        - strong [ref=e28]: Create runs
        - text: from text or input files, selecting model and filtering parameters.
      - listitem [ref=e29]:
        - strong [ref=e30]: Track progress
        - text: "through pipeline stages: ingest → generate → validate → filter → evaluate."
      - listitem [ref=e31]:
        - strong [ref=e32]: View results
        - text: details and logs for each run, with error and success insights.
      - listitem [ref=e33]:
        - strong [ref=e34]: Cancel jobs
        - text: that are running, with immediate state feedback.
  - generic [ref=e35]:
    - generic [ref=e36]:
      - heading "Create Run" [level=2] [ref=e37]
      - generic [ref=e38]:
        - generic [ref=e39]:
          - text: Input text
          - textbox "Input text" [ref=e40]:
            - /placeholder: Paste text sample to ingest
        - generic [ref=e41]:
          - text: Input path
          - textbox "Input path" [ref=e42]:
            - /placeholder: /data/sample_inputs/sample_text.txt
        - generic [ref=e43]:
          - generic [ref=e44]:
            - text: Mock LLM
            - combobox "Mock LLM" [ref=e45]:
              - option "true" [selected]
              - option "false"
          - generic [ref=e46]:
            - text: Agent
            - combobox "Agent" [ref=e47]:
              - option "false" [selected]
              - option "true"
        - generic [ref=e48]:
          - generic [ref=e49]:
            - text: Workers
            - spinbutton "Workers" [ref=e50]: "1"
          - generic [ref=e51]:
            - text: Steering
            - combobox "Steering" [ref=e52]:
              - option "auto" [selected]
              - option "review-low"
              - option "review-all"
        - generic [ref=e53]:
          - text: Threshold 0.7
          - slider "Threshold 0.7" [ref=e54]: "0.7"
        - button "Start run" [ref=e55] [cursor=pointer]
        - paragraph
    - generic [ref=e56]:
      - generic [ref=e57]:
        - heading "Run List" [level=2] [ref=e58]
        - generic [ref=e59]:
          - combobox [ref=e60]:
            - option "All" [selected]
            - option "Pending"
            - option "Running"
            - option "Succeeded"
            - option "Failed"
            - option "Canceled"
          - button "Refresh" [ref=e61] [cursor=pointer]
      - table [ref=e63]:
        - rowgroup [ref=e64]:
          - row "Run Status Updated Error Action" [ref=e65]:
            - columnheader "Run" [ref=e66]
            - columnheader "Status" [ref=e67]
            - columnheader "Updated" [ref=e68]
            - columnheader "Error" [ref=e69]
            - columnheader "Action" [ref=e70]
        - rowgroup
    - generic [ref=e71]:
      - heading "Selected Run Details" [level=2] [ref=e72]
      - paragraph [ref=e73]: Select a run from the list to view details and logs.
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test';
  2  | 
  3  | test('homepage has title and create-run form', async ({ page }) => {
  4  |   await page.goto('/');
> 5  |   await expect(page.getByRole('heading', { name: 'synthDataLab Production UI' })).toBeVisible();
     |                                                                                   ^ Error: expect(locator).toBeVisible() failed
  6  |   await expect(page.getByRole('heading', { name: 'Create run' })).toBeVisible();
  7  |   await expect(page.getByRole('button', { name: 'Start run' })).toBeVisible();
  8  | });
  9  | 
  10 | // A simple integration smoke test that uses mocked network response. This does not require backend.
  11 | test('run creation shows error when no input is provided', async ({ page }) => {
  12 |   await page.goto('/');
  13 |   await page.getByRole('button', { name: 'Start run' }).click();
  14 |   await expect(page.getByText('Please provide input text or input path.')).toBeVisible();
  15 | });
  16 | 
```