# Component Specifications

## Button Component

### Variants

| Variant | Use Case | Visual Weight |
|---------|----------|---------------|
| Primary | Main actions (Submit, Confirm) | High |
| Secondary | Cancel, Clear, Back | Medium |
| Destructive | Delete, Remove, Archive | High (danger) |

### States

| State | Primary | Secondary | Destructive |
|-------|---------|-----------|-------------|
| **Default** | `--color-interactive-primary` bg, white text | Transparent bg, muted text, border | `--color-interactive-destructive` bg, white text |
| **Hover** | Darker orange (`--color-interactive-primary-hover`) | `--color-zinc-700` bg, white text | Darker red (`--color-interactive-destructive-hover`) |
| **Active** | Darkest orange (`--color-interactive-primary-active`) | `--color-zinc-600` bg | Darkest red (`--color-interactive-destructive-active`) |
| **Disabled** | `--color-interactive-primary-disabled` bg, 50% opacity | `--color-zinc-800` bg, 50% opacity | `--color-interactive-primary-disabled` bg, 50% opacity |
| **Focus** | 3px orange ring (`--shadow-focus`) | 3px orange ring | 3px orange ring |

### Dimensions

| Property | Value | Notes |
|----------|-------|-------|
| Height | 44px | Meets touch target minimum |
| Padding | 12px 12px | --space-3 horizontal, --space-2 vertical |
| Border Radius | 8px | --radius-md |
| Font Size | 15px | --font-size-md |
| Font Weight | 700 | --font-weight-bold |

### Accessibility

- Minimum touch target: 44x44px ✓
- Focus ring visible on keyboard navigation ✓
- Disabled state: `cursor: not-allowed`, 50% opacity, `aria-disabled="true"`
- Loading state: Show spinner, disable button, maintain width

---

## Input Component

### States

| State | Visual Treatment |
|-------|------------------|
| **Default** | `--color-background-secondary` bg, `--color-border-default` border |
| **Focus** | `--color-border-focus` border, orange ring (`--shadow-focus`) |
| **Error** | `--color-status-error-border` border, error message below |
| **Disabled** | `--color-background-tertiary` bg, 50% opacity, `cursor: not-allowed` |

### Dimensions

| Property | Value | Notes |
|----------|-------|-------|
| Height | 44px | Meets touch target minimum |
| Padding | 0 12px | --space-3 horizontal |
| Border Radius | 8px | --radius-md |
| Font Size | 15px | --font-size-md |
| Width | 100% | Full container width |

### Labels

- Always visible (not placeholder-only)
- Position: Above input
- Font: 14px semibold, `--color-foreground-secondary`
- Required indicator: Asterisk (*) after label text

### Error Handling

- Show error below input
- Link with `aria-describedby`
- Use `--color-status-error-*` tokens
- Include recovery guidance

---

## Card Component

### Usage

- Auth panel (login/register)
- Content containers
- Feature sections

### States

| Property | Value |
|----------|-------|
| Background | `--color-background-secondary` |
| Border | 1px solid `--color-border-default` |
| Border Radius | 12px (`--radius-lg`) |
| Padding | 28px (`--spacing-section-padding-x`) |
| Shadow | Combined (sm + md) |

### Elevation Levels

| Level | Shadow | Use Case |
|-------|--------|----------|
| 0 | None | Flat cards, inline content |
| 1 | `--shadow-sm` | Cards in lists |
| 2 | `--shadow-combined` | Elevated cards (auth panel) |

---

## Status/Alert Component

### Variants

| Variant | Background | Text | Border | Use Case |
|---------|------------|------|--------|----------|
| Success | `--color-green-900` | `--color-green-300` | `--color-green-700` | Form success, completion |
| Error | `--color-red-900` | `--color-red-200` | `--color-red-700` | Validation errors, failures |
| Warning | `--color-orange-900` | `--color-orange-300` | `--color-orange-700` | Caution, degraded state |
| Info | `--color-slate-800` | `--color-slate-200` | `--color-slate-600` | Information, hints |

### Dimensions

| Property | Value |
|----------|-------|
| Padding | 8px 12px |
| Border Radius | 8px |
| Font Size | 14px |
| Line Height | 1.5 |

---

## Link Component

### States

| State | Color | Decoration |
|-------|-------|------------|
| Default | `--color-interactive-primary` | None |
| Hover | `--color-orange-300` | None |
| Visited | `--color-orange-400` | None |

### Properties

| Property | Value |
|----------|-------|
| Font Weight | 600 (semibold) |
| Font Size | 14px |
| Transition | Color 0.15s |

---

## Typography Scale

### Headings

| Level | Size | Weight | Line Height | Color |
|-------|------|--------|-------------|-------|
| Brand (H1) | 22px | 700 | 1.35 | `--color-foreground-primary` |
| Section (H2) | 20px | 700 | 1.4 | `--color-foreground-primary` |
| Subsection (H3) | 18px | 600 | 1.4 | `--color-foreground-primary` |

### Body Text

| Style | Size | Weight | Line Height | Color |
|-------|------|--------|-------------|-------|
| Default | 15px | 400 | 1.5 | `--color-foreground-primary` |
| Secondary | 14px | 400 | 1.6 | `--color-foreground-muted` |
| Label | 14px | 600 | 1.5 | `--color-foreground-secondary` |
| Caption | 13px | 400 | 1.6 | `--color-foreground-muted` |

---

## Spacing Scale

### Component Level

| Token | Value | Usage |
|-------|-------|-------|
| `--spacing-component-padding-x` | 12px | Button/input horizontal padding |
| `--spacing-component-padding-y` | 8px | Button/input vertical padding |
| `--spacing-component-gap` | 8px | Gap between related elements |

### Section Level

| Token | Value | Usage |
|-------|-------|-------|
| `--spacing-section-padding-x` | 28px | Card padding |
| `--spacing-section-padding-y` | 28px | Card padding |
| `--spacing-section-gap` | 20px | Gap between sections |

### Page Level

| Token | Value | Usage |
|-------|-------|-------|
| `--spacing-page-padding-x` | 32px | Page margins |
| `--spacing-page-padding-y` | 32px | Page margins |
| `--spacing-page-max-width` | 440px | Max content width |

---

## Border Radius Scale

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-none` | 0 | No radius |
| `--radius-sm` | 6px | Small elements (badges, tags) |
| `--radius-md` | 8px | Buttons, inputs, status messages |
| `--radius-lg` | 12px | Cards, panels |
| `--radius-full` | 999px | Pills, circular elements |

---

## Animation & Transitions

### Timing Tokens

| Token | Value | Usage |
|-------|-------|-------|
| `--transition-fast` | 0.15s | Hover states, focus rings |
| `--transition-normal` | 0.2s | Standard transitions |
| `--transition-slow` | 0.3s | Complex animations |

### Easing

- **Default:** `ease-in-out`
- **Enter:** `ease-out` (decelerate)
- **Exit:** `ease-in` (accelerate)

### Reduced Motion

Respect `prefers-reduced-motion` media query:
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## Accessibility Checklist

### Touch Targets

- [ ] All interactive elements ≥ 44x44px
- [ ] Minimum 8px gap between touch targets
- [ ] Expanded hit area for small icons

### Focus Management

- [ ] Visible focus ring on all interactive elements
- [ ] Focus order matches visual order
- [ ] Focus not obscured by sticky headers/overlays

### Color & Contrast

- [ ] Text contrast ≥ 4.5:1 (normal text)
- [ ] Text contrast ≥ 3:1 (large text)
- [ ] Non-text contrast ≥ 3:1 (icons, borders)
- [ ] Color not sole indicator of state

### Keyboard Navigation

- [ ] All functionality accessible via keyboard
- [ ] Tab order logical and predictable
- [ ] Escape closes modals/dropdowns

### Screen Readers

- [ ] All images have alt text
- [ ] Form fields have labels
- [ ] Errors linked to fields with `aria-describedby`
- [ ] Status messages use `aria-live`
