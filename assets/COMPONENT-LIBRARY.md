# Component Library

A complete reference for all UI components in the FixOldImg project.

## Table of Contents

1. [Design Tokens](#design-tokens)
2. [Button Component](#button-component)
3. [Input Component](#input-component)
4. [Card Component](#card-component)
5. [Status Component](#status-component)
6. [Link Component](#link-component)
7. [Typography](#typography)
8. [Spacing](#spacing)
9. [Accessibility](#accessibility)

---

## Design Tokens

### Token Architecture

```
Primitive (Raw Values)
    ↓
Semantic (Purpose Aliases)
    ↓
Component (Component-Specific)
```

### Token Categories

| Category | Examples | Usage |
|----------|----------|-------|
| Colors | `--color-orange-600` | Raw color values |
| Spacing | `--space-4` | Consistent spacing scale |
| Typography | `--font-size-md` | Font sizes and weights |
| Border Radius | `--radius-md` | Rounded corners |
| Shadows | `--shadow-focus` | Elevation and focus states |
| Transitions | `--transition-fast` | Animation timing |

### Token Usage

```css
/* ✅ Correct - Use semantic tokens */
background: var(--color-interactive);
color: var(--color-fg-primary);
padding: var(--space-4);

/* ❌ Wrong - Don't use raw values */
background: #ea580c;
color: #f4f4f5;
padding: 16px;
```

---

## Button Component

### Variants

#### Primary Button
For main actions (Submit, Confirm, Save).

```html
<button class="button-primary">开始修复</button>
```

```css
.button-primary {
  background: var(--color-interactive);
  color: var(--color-white);
  border: 1px solid var(--color-interactive);
  border-radius: var(--radius-md);
  height: 44px;
  padding: var(--space-2) var(--space-3);
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-bold);
  cursor: pointer;
  transition: all var(--transition-fast);
}
```

**States:**

| State | Visual Treatment |
|-------|------------------|
| Default | Orange background, white text |
| Hover | Darker orange (`--color-interactive-hover`) |
| Active | Darkest orange (`--color-interactive-active`) |
| Disabled | Gray background, 50% opacity, `cursor: not-allowed` |
| Focus | 3px orange ring (`--shadow-focus`) |

#### Secondary Button
For cancel, clear, or back actions.

```html
<button class="button-secondary">清空图片</button>
```

```css
.button-secondary {
  background: transparent;
  color: var(--color-fg-muted);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  height: 44px;
  padding: var(--space-2) var(--space-3);
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-fast);
}
```

**States:**

| State | Visual Treatment |
|-------|------------------|
| Default | Transparent background, muted text |
| Hover | Gray background, white text |
| Active | Darker gray background |
| Focus | 3px orange ring |

#### Destructive Button
For delete, remove, or archive actions.

```html
<button class="button-destructive">删除用户</button>
```

```css
.button-destructive {
  background: var(--color-error-bg);
  color: var(--color-white);
  border: 1px solid var(--color-error-border);
  border-radius: var(--radius-md);
  height: 44px;
  padding: var(--space-2) var(--space-3);
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-bold);
  cursor: pointer;
  transition: all var(--transition-fast);
}
```

**States:**

| State | Visual Treatment |
|-------|------------------|
| Default | Red background, white text |
| Hover | Darker red |
| Active | Darkest red |
| Focus | 3px orange ring |

### Button Sizes

| Size | Height | Padding | Font Size | Use Case |
|------|--------|---------|-----------|----------|
| Small | 32px | 4px 8px | 12px | Inline actions |
| Default | 44px | 8px 12px | 15px | Standard buttons |
| Large | 56px | 12px 16px | 18px | Primary CTAs |

### Button with Icon

```html
<button class="button-primary">
  <svg class="icon">...</svg>
  开始修复
</button>
```

```css
.button-primary .icon {
  width: 16px;
  height: 16px;
  margin-right: var(--space-2);
}
```

### Loading State

```html
<button class="button-primary" disabled>
  <svg class="spinner">...</svg>
  处理中...
</button>
```

```css
.button-primary:disabled {
  background: var(--color-interactive-disabled);
  border-color: var(--color-interactive-disabled);
  cursor: not-allowed;
  opacity: 0.5;
}

.spinner {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
```

---

## Input Component

### Basic Input

```html
<input class="input" type="text" placeholder="请输入用户名">
```

```css
.input {
  width: 100%;
  height: 44px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg-secondary);
  color: var(--color-fg-primary);
  font-size: var(--font-size-md);
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
}
```

### Input with Label

```html
<div class="form-group">
  <label class="form-label" for="username">用户名</label>
  <input class="input" id="username" type="text" required>
</div>
```

```css
.form-group {
  margin-bottom: var(--space-3);
}

.form-label {
  display: block;
  margin-bottom: var(--space-1);
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-fg-secondary);
}
```

### Input States

| State | Visual Treatment |
|-------|------------------|
| Default | Dark background, subtle border |
| Focus | Orange border, orange ring |
| Error | Red border, error message below |
| Disabled | Muted background, 50% opacity |
| Success | Green border (optional) |

#### Focus State

```css
.input:focus {
  outline: none;
  border-color: var(--color-border-focus);
  box-shadow: var(--shadow-focus);
}
```

#### Error State

```html
<div class="form-group">
  <label class="form-label" for="password">密码</label>
  <input class="input input-error" id="password" type="password" aria-describedby="password-error">
  <span class="form-error" id="password-error">密码长度不能少于 6 位</span>
</div>
```

```css
.input-error {
  border-color: var(--color-error-border);
}

.form-error {
  display: block;
  margin-top: var(--space-1);
  font-size: var(--font-size-sm);
  color: var(--color-error-fg);
}
```

#### Disabled State

```css
.input:disabled {
  background: var(--color-bg-tertiary);
  cursor: not-allowed;
  opacity: 0.5;
}
```

### Input Types

| Type | Usage | Keyboard |
|------|-------|----------|
| `text` | General text input | Default |
| `password` | Password fields | Default |
| `email` | Email addresses | Email keyboard |
| `number` | Numeric values | Number keyboard |
| `tel` | Phone numbers | Phone keyboard |

### Input Sizes

| Size | Height | Padding | Font Size | Use Case |
|------|--------|---------|-----------|----------|
| Small | 32px | 0 8px | 13px | Compact forms |
| Default | 44px | 0 12px | 15px | Standard forms |
| Large | 56px | 0 16px | 18px | Search, hero inputs |

---

## Card Component

### Basic Card

```html
<div class="card">
  <h2 class="card-title">用户注册</h2>
  <form>...</form>
</div>
```

```css
.card {
  width: 100%;
  padding: var(--space-7);
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-combined);
}
```

### Card Variants

#### Elevated Card
For primary content areas (auth panel).

```css
.card-elevated {
  box-shadow: var(--shadow-combined);
}
```

#### Flat Card
For inline content (status messages).

```css
.card-flat {
  box-shadow: none;
}
```

#### Outlined Card
For selectable items.

```css
.card-outlined {
  box-shadow: none;
  border: 2px solid var(--color-border);
}

.card-outlined:hover {
  border-color: var(--color-interactive);
}
```

### Card States

| State | Visual Treatment |
|-------|------------------|
| Default | Standard shadow and border |
| Hover | Slightly elevated (optional) |
| Selected | Highlighted border |
| Disabled | Muted background |

---

## Status Component

### Status Variants

#### Success

```html
<div class="status status-success">
  注册成功，请返回登录页登录
</div>
```

```css
.status-success {
  background: var(--color-success-bg);
  color: var(--color-success-fg);
  border: 1px solid var(--color-success-border);
}
```

#### Error

```html
<div class="status status-error">
  用户名格式不正确
</div>
```

```css
.status-error {
  background: var(--color-error-bg);
  color: var(--color-error-fg);
  border: 1px solid var(--color-error-border);
}
```

#### Warning

```html
<div class="status status-warning">
  ⚠️ 本次未检测到人脸，已跳过面部增强
</div>
```

```css
.status-warning {
  background: var(--color-warning-bg);
  color: var(--color-warning-fg);
  border: 1px solid var(--color-warning-border);
}
```

#### Info

```html
<div class="status status-info">
  首次调用需要加载 DDColor 权重（约 3 秒）
</div>
```

```css
.status-info {
  background: var(--color-bg-tertiary);
  color: var(--color-fg-secondary);
  border: 1px solid var(--color-border);
}
```

### Status Base Styles

```css
.status {
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  font-size: var(--font-size-base);
  line-height: var(--line-height-normal);
}
```

### Status with Icon

```html
<div class="status status-success">
  <svg class="status-icon">✓</svg>
  <span>注册成功</span>
</div>
```

```css
.status-icon {
  width: 16px;
  height: 16px;
  margin-right: var(--space-2);
}
```

---

## Link Component

### Basic Link

```html
<a class="link" href="/register">立即注册</a>
```

```css
.link {
  color: var(--color-interactive);
  text-decoration: none;
  font-weight: var(--font-weight-semibold);
  font-size: var(--font-size-base);
  transition: color var(--transition-fast);
}

.link:hover {
  color: var(--color-orange-300);
}

.link:focus-visible {
  outline: 2px solid var(--color-border-focus);
  outline-offset: 2px;
}
```

### Link Variants

#### Inline Link
For text links within paragraphs.

```css
.link-inline {
  text-decoration: underline;
}
```

#### Button Link
For navigation that looks like a button.

```css
.link-button {
  display: inline-block;
  padding: var(--space-2) var(--space-4);
  background: var(--color-interactive);
  color: var(--color-white);
  border-radius: var(--radius-md);
}
```

---

## Typography

### Font Scale

| Token | Size | Usage |
|-------|------|-------|
| `--font-size-xs` | 12px | Captions, helper text |
| `--font-size-sm` | 13px | Small text, labels |
| `--font-size-base` | 14px | Body text, labels |
| `--font-size-md` | 15px | Default body text |
| `--font-size-lg` | 18px | Subheadings |
| `--font-size-xl` | 20px | Section headings |
| `--font-size-2xl` | 22px | Page titles |

### Font Weights

| Token | Weight | Usage |
|-------|--------|-------|
| `--font-weight-normal` | 400 | Body text |
| `--font-weight-medium` | 500 | Labels, emphasis |
| `--font-weight-semibold` | 600 | Labels, buttons |
| `--font-weight-bold` | 700 | Headings, buttons |

### Line Heights

| Token | Value | Usage |
|-------|-------|-------|
| `--line-height-tight` | 1.35 | Headings |
| `--line-height-snug` | 1.4 | Subheadings |
| `--line-height-normal` | 1.5 | Body text |
| `--line-height-relaxed` | 1.6 | Long-form text |

### Heading Styles

```css
.heading-brand {
  font-size: var(--font-size-2xl);
  font-weight: var(--font-weight-bold);
  line-height: var(--line-height-tight);
  color: var(--color-fg-primary);
}

.heading-section {
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-bold);
  line-height: var(--line-height-snug);
  color: var(--color-fg-primary);
}

.heading-subsection {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-snug);
  color: var(--color-fg-primary);
}
```

### Body Text Styles

```css
.body-default {
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-normal);
  color: var(--color-fg-primary);
}

.body-secondary {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-relaxed);
  color: var(--color-fg-muted);
}

.caption {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-normal);
  line-height: var(--line-height-relaxed);
  color: var(--color-fg-muted);
}
```

---

## Spacing

### Spacing Scale

| Token | Value | Usage |
|-------|-------|-------|
| `--space-0` | 0px | No spacing |
| `--space-1` | 4px | Tight spacing |
| `--space-2` | 8px | Small spacing |
| `--space-3` | 12px | Default spacing |
| `--space-4` | 16px | Medium spacing |
| `--space-5` | 20px | Section spacing |
| `--space-6` | 24px | Large spacing |
| `--space-7` | 28px | Card padding |
| `--space-8` | 32px | Page margins |

### Spacing Patterns

#### Component Spacing
```css
.component {
  padding: var(--space-2) var(--space-3);
  margin-bottom: var(--space-3);
}
```

#### Section Spacing
```css
.section {
  padding: var(--space-7);
  margin-bottom: var(--space-5);
}
```

#### Page Spacing
```css
.page {
  padding: var(--space-8);
  max-width: 440px;
}
```

---

## Accessibility

### Touch Targets

All interactive elements must be at least **44x44px**:

```css
.button-primary,
.button-secondary,
.input {
  height: 44px;
  min-width: 44px;
}
```

### Focus Management

Visible focus rings for keyboard navigation:

```css
:focus-visible {
  outline: 2px solid var(--color-border-focus);
  outline-offset: 2px;
}
```

### Color Contrast

| Text Type | Minimum Ratio | Example |
|-----------|---------------|---------|
| Normal text | 4.5:1 | Body text, labels |
| Large text | 3:1 | Headings, buttons |
| Non-text | 3:1 | Icons, borders |

### Reduced Motion

Respect user preferences:

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

### Screen Readers

- All images need `alt` text
- Form fields need `label` elements
- Errors linked with `aria-describedby`
- Status messages use `aria-live`

### Keyboard Navigation

- Tab order matches visual order
- All functionality accessible via keyboard
- Escape closes modals/dropdowns
- No keyboard traps

---

## Dark/Light Mode

### Token-Based Theming

```css
/* Dark Mode (Default) */
:root {
  --color-bg-primary: #020617;
  --color-fg-primary: #f4f4f5;
}

/* Light Mode */
[data-theme="light"] {
  --color-bg-primary: #ffffff;
  --color-fg-primary: #0f172a;
}
```

### Usage

```html
<html data-theme="dark">
  <!-- Dark mode content -->
</html>

<html data-theme="light">
  <!-- Light mode content -->
</html>
```

### JavaScript Toggle

```javascript
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
}
```

---

## Best Practices

1. **Always use tokens** - Never hardcode colors, spacing, or typography
2. **Mobile-first** - Design for small screens first
3. **Accessible** - Meet WCAG 2.1 AA standards
4. **Consistent** - Use the same patterns throughout
5. **Responsive** - Test on 375px, 768px, 1024px
6. **Performant** - Minimize reflows, use transforms
7. **Maintainable** - Document all components and tokens
