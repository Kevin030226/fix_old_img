# Migration Guide: Hardcoded CSS → Design Tokens

## Overview

This guide shows how to migrate the current hardcoded CSS values in `main.py` to use the new design token system.

## Current State (Problems)

### 1. Hardcoded Colors

**Before:**
```python
"background:#0f0f11!important;"
"color:#f4f4f5;"
"border:1px solid #3f3f46;"
"background:#ea580c;"
```

**After:**
```python
"background:var(--color-background-primary)!important;"
"color:var(--color-foreground-primary);"
"border:1px solid var(--color-border-default);"
"background:var(--color-interactive-primary);"
```

### 2. Hardcoded Spacing

**Before:**
```python
"padding:28px;"
"margin:14px 0 6px;"
"padding:0 12px;"
```

**After:**
```python
"padding:var(--spacing-section-padding-x);"
"margin:var(--space-3) 0 var(--space-1);"
"padding:0 var(--spacing-component-padding-x);"
```

### 3. Hardcoded Typography

**Before:**
```python
"font-size:22px;"
"font-weight:700;"
"line-height:1.35;"
```

**After:**
```python
"font-size:var(--typography-heading-brand-size);"
"font-weight:var(--typography-heading-brand-weight);"
"line-height:var(--typography-heading-brand-line-height);"
```

### 4. Hardcoded Border Radius

**Before:**
```python
"border-radius:12px;"
"border-radius:8px;"
"border-radius:999px;"
```

**After:**
```python
"border-radius:var(--radius-lg);"
"border-radius:var(--radius-md);"
"border-radius:var(--radius-full);"
```

### 5. Hardcoded Shadows

**Before:**
```python
"box-shadow:0 1px 2px rgba(0,0,0,.3),0 16px 40px rgba(0,0,0,.45);"
"box-shadow:0 0 0 3px rgba(251,146,60,.25);"
```

**After:**
```python
"box-shadow:var(--shadow-combined);"
"box-shadow:var(--shadow-focus);"
```

---

## Step-by-Step Migration

### Step 1: Add Token Import

Add to `main.py` or create a separate CSS file:

```python
# Option 1: Inline CSS variable definitions (recommended for Gradio)
TOKEN_CSS = """
:root {
  /* Import from design-tokens.css */
  --color-background-primary: #020617;
  --color-background-secondary: #27272a;
  --color-foreground-primary: #f4f4f5;
  /* ... all other tokens ... */
}
"""

# Option 2: External CSS file (if Gradio supports it)
# Add to Gradio Blocks: gr.Blocks(css="assets/design-tokens.css")
```

### Step 2: Update `build_auth_page_css()`

**Current:**
```python
def build_auth_page_css(center_body=True):
    return (
        "<style>"
        "html{color-scheme:dark}"
        "*{box-sizing:border-box}"
        "html,body{min-height:100%;margin:0}"
        "body{"
        f"{body_layout}"
        "background:#0f0f11!important;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
        "color:#f4f4f5;"
        "}"
        ".auth-page{width:min(440px,calc(100vw - 32px));margin:0 auto;padding:32px 0}"
        ".auth-brand{margin:0 0 20px;text-align:center;font-size:22px;line-height:1.35;font-weight:700;color:#ffffff}"
        ".auth-panel{width:100%;padding:28px;background:#27272a;border:1px solid #3f3f46;border-radius:12px;"
        "box-shadow:0 1px 2px rgba(0,0,0,.3),0 16px 40px rgba(0,0,0,.45)}"
        # ... more hardcoded values ...
        "</style>"
    )
```

**After:**
```python
def build_auth_page_css(center_body=True):
    body_layout = (
        "display:flex!important;flex-direction:column!important;"
        "justify-content:center!important;align-items:center!important;"
        if center_body
        else ""
    )
    return (
        "<style>"
        "html{color-scheme:dark}"
        "*{box-sizing:border-box}"
        "html,body{min-height:100%;margin:0}"
        "body{"
        f"{body_layout}"
        "background:var(--color-background-primary)!important;"
        "font-family:var(--font-family);"
        "color:var(--color-foreground-primary);"
        "}"
        ".auth-page{width:min(var(--spacing-page-max-width),calc(100vw - 32px));margin:0 auto;padding:var(--spacing-page-padding-y) 0}"
        ".auth-brand{margin:0 0 var(--spacing-section-gap);text-align:center;font-size:var(--typography-heading-brand-size);line-height:var(--typography-heading-brand-line-height);font-weight:var(--typography-heading-brand-weight);color:var(--color-foreground-primary)}"
        ".auth-panel{width:100%;padding:var(--spacing-section-padding-x);background:var(--color-background-secondary);border:1px solid var(--color-border-default);border-radius:var(--radius-lg);box-shadow:var(--shadow-combined)}"
        # ... more token-based values ...
        "</style>"
    )
```

### Step 3: Update Button Styles

**Current:**
```python
".auth-button{width:100%;height:42px;margin-top:20px;border:1px solid #ea580c;border-radius:8px;"
"background:#ea580c;color:#ffffff;font-size:15px;"
"font-weight:700;cursor:pointer;transition:filter .15s,border-color .15s}"
".auth-button:hover{background:#c2410c;border-color:#c2410c}"
```

**After:**
```python
".auth-button{width:100%;height:44px;margin-top:var(--spacing-section-gap);border:1px solid var(--color-interactive-primary);border-radius:var(--radius-md);"
"background:var(--color-interactive-primary);color:var(--color-foreground-primary);font-size:var(--typography-body-size);"
"font-weight:var(--font-weight-bold);cursor:pointer;transition:all var(--transition-fast)}"
".auth-button:hover{background:var(--color-interactive-primary-hover);border-color:var(--color-interactive-primary-hover)}"
".auth-button:active{background:var(--color-interactive-primary-active);border-color:var(--color-interactive-primary-active)}"
".auth-button:focus{outline:none;box-shadow:var(--shadow-focus)}"
".auth-button:disabled{background:var(--color-interactive-primary-disabled);border-color:var(--color-interactive-primary-disabled);cursor:not-allowed;opacity:0.5}"
```

### Step 4: Update Input Styles

**Current:**
```python
".auth-input{width:100%;height:42px;padding:0 12px;border:1px solid #3f3f46;border-radius:8px;"
"background:#27272a;font-size:15px;color:#f4f4f5;transition:border-color .15s,box-shadow .15s}"
".auth-input::placeholder{color:#71717a}"
".auth-input:focus{outline:none;border-color:#fb923c;box-shadow:0 0 0 3px rgba(251,146,60,.25)}"
```

**After:**
```python
".auth-input{width:100%;height:44px;padding:0 var(--spacing-component-padding-x);border:1px solid var(--color-border-default);border-radius:var(--radius-md);"
"background:var(--color-background-secondary);font-size:var(--typography-body-size);color:var(--color-foreground-primary);transition:border-color var(--transition-fast),box-shadow var(--transition-fast)}"
".auth-input::placeholder{color:var(--color-foreground-muted)}"
".auth-input:focus{outline:none;border-color:var(--color-border-focus);box-shadow:var(--shadow-focus)}"
".auth-input:disabled{background:var(--color-background-tertiary);cursor:not-allowed;opacity:0.5}"
```

### Step 5: Update Status Messages

**Current:**
```python
".auth-status.success{background:#14532d;color:#86efac}"
".auth-status.error{background:#7f1d1d;color:#fecaca}"
```

**After:**
```python
".auth-status.success{background:var(--color-status-success-bg);color:var(--color-status-success-fg);border:1px solid var(--color-status-success-border)}"
".auth-status.error{background:var(--color-status-error-bg);color:var(--color-status-error-fg);border:1px solid var(--color-status-error-border)}"
```

### Step 6: Update Links

**Current:**
```python
".auth-link{display:block;margin-top:16px;text-align:center;font-size:14px;color:#fb923c;"
"text-decoration:none;font-weight:600}"
".auth-link:hover{color:#fdba74}"
```

**After:**
```python
".auth-link{display:block;margin-top:var(--spacing-section-gap);text-align:center;font-size:var(--typography-body-secondary-size);color:var(--color-interactive-primary);"
"text-decoration:none;font-weight:var(--font-weight-semibold);transition:color var(--transition-fast)}"
".auth-link:hover{color:var(--color-orange-300)}"
```

---

## Benefits of Migration

### 1. Consistency
- Single source of truth for all design values
- Changes propagate automatically across components

### 2. Maintainability
- Update one token → all usages update
- Easy to document and understand

### 3. Theme Support
- Enable light mode by redefining semantic tokens
- Support high contrast mode
- Support custom brand themes

### 4. Accessibility
- Enforce consistent contrast ratios
- Ensure minimum touch targets
- Standardize focus states

### 5. Performance
- CSS variables are efficient (cached by browser)
- No runtime computation needed

---

## Testing Checklist

After migration, verify:

- [ ] All colors render correctly
- [ ] All spacing is consistent
- [ ] All typography scales properly
- [ ] Focus rings visible on keyboard navigation
- [ ] Touch targets ≥ 44x44px
- [ ] Status messages have proper contrast
- [ ] Hover/active states work
- [ ] Disabled states are visually clear
- [ ] Reduced motion works (test with `prefers-reduced-motion: reduce`)
- [ ] Responsive design works (375px, 768px, 1024px)

---

## Rollback Plan

If issues arise:

1. Keep original hardcoded values as comments
2. Use feature flags to toggle between token and hardcoded versions
3. Test thoroughly before full deployment
