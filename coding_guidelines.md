# React Coding Guidelines

These guidelines focus on logic modularity, robustness, naming, and effective use of XState. Styling and type strictness are secondary.

---

## 1. Component Design

### ✅ Do:
- Keep components focused on UI rendering.
- Delegate logic and state management to hooks or state machines.
- Pass only the props a component needs.
- Use functional components with named functions (avoid anonymous default exports).

### ❌ Avoid:
- Logic-heavy components.
- Deep prop drilling.
- Components that manage their own state transitions outside of machines.

---

## 2. Functionality and Logic Modularity

### ✅ Do:
- Organize files **by feature**, not by type.
- Use folders like `features/checkout/Checkout.tsx` with co-located `machine.ts`, `useCheckout.ts`.
- Keep API calls, side-effects, and derived logic in services/hooks.
- Compose logic using pure functions or machine actions/guards.

### ❌ Avoid:
- Copy-pasting logic.
- Creating giant `utils.ts` files with random logic.
- Implicit side-effects in shared code.

---

## 3. Null Checks and Defensive Coding

### ✅ Do:
- Use optional chaining (`?.`) and nullish coalescing (`??`) where applicable.
- Always verify existence of fetched data or optional props.
- Render fallback components for empty/null values.

```tsx
{user?.name ?? 'Guest'}
{items?.length ? <ItemList items={items} /> : <EmptyState />}
```

### ❌ Avoid:
- Assuming props or API data are always defined.
- Rendering deeply nested objects without guards.

---

## 4. XState Usage

### ✅ Do:
- Write one `machine.ts` per feature. Name machines clearly (`checkoutMachine`, `profileMachine`).
- Handle all transitions, guards, and effects inside the machine.
- Use clear and descriptive event names (e.g., `SUBMIT_FORM`, `CANCEL_PAYMENT`, `RETRY_FETCH`).
- Store machine files under the corresponding feature directory.

### ❌ Avoid:
- Defining logic or transitions in React components.
- Triggering machine events without clear action mapping.
- Using vague event names like `NEXT` or `CLICKED`.

---

## 5. Naming Conventions

Use **consistent, descriptive, and unambiguous** names across components, files, states, and events.

### Files and Folders
- `features/checkout/Checkout.tsx`
- `features/checkout/machine.ts`
- `features/checkout/useCheckout.ts`
- `services/checkoutService.ts`
- `components/Button.tsx`

### State Machines
- `checkoutMachine`, `authMachine`, `formMachine`
- States: `idle`, `loading`, `success`, `error`
- Events: `SUBMIT`, `RETRY`, `CANCEL`, `CONFIRM_PAYMENT`

### Context Variables
- `userData`, `formValues`, `cartItems`, not `data`, `temp`, `stuff`
- `setUserData`, `updateCartItems`, not `changeIt`, `doThing`

### Custom Hooks
- Always prefix with `use`: `useAuth`, `useCartSync`, `useFormErrors`

### Component Names
- Always PascalCase: `UserProfile`, `CartItem`, `LoginForm`
- Avoid abbreviations: prefer `OrderSummary` over `OrdSum`

### Services / Helpers
- Verb-based and intent-driven: `fetchUserProfile()`, `formatCurrency()`, `retryPayment()`

---

## 6. Cross-Feature Safety

### ✅ Do:
- Coordinate changes to shared files or global state types.
- Keep interfaces for shared types or events versioned or documented.
- Clearly type events and context to avoid accidental breakages.

### ❌ Avoid:
- Silent changes to helper logic or types that are used in other features.
- Shared mutable state between machines.

---

## 7. Project Structure (Recommended)

```
/src
  /features
    /checkout
      Checkout.tsx
      machine.ts
      useCheckout.ts
    /profile
      Profile.tsx
      machine.ts
  /components
  /hooks
  /services
  /utils
```

---

## 8. Merge Checklist ✅

- [ ] Is all logic in the machine, not the component?
- [ ] Are all null/undefined cases handled?
- [ ] Did you verify impact on other features?
- [ ] Are naming conventions followed?
- [ ] Are new helpers, types, or events clearly scoped?

---

## 9. Testing and Debugging

### ✅ Do:
- Use `devTools: true` in XState for development.
- Write test cases for state transitions, guards, and actions.
- Test machines in isolation where possible.

---

## 10. Summary

- **Keep components dumb. Keep machines smart.**
- **Name things clearly and consistently.**
- **Check before you break other features.**
- **Never assume data is present.**
- **Let the state machine control the flow.**