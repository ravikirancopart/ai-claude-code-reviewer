# Vue Component Guidelines

## Amplitude Event Properties
- All Amplitude event properties MUST use camelCase naming
- Example correct: `pageLocation`, `userType`, `accountStatus`
- Example incorrect: `page_location`, `user-type`, `account-status`
- When tracking page views, use `location` as the parameter name
- Common properties: `location`, `pageType`, `userId`, `accountType`

## Testing Requirements
- Every Vue component MUST have a corresponding test file
- Test files MUST use the `.spec.ts` extension
- Test files MUST be in the same directory as the component
- Example: For `UserProfile.vue`, create `UserProfile.spec.ts`
- Tests MUST cover component props, events, and main functionality

## CSS Methodology
- MUST use BEM (Block Element Modifier) methodology
- Block: Standalone entity (e.g., `user-profile`)
- Element: Part of block (e.g., `user-profile__avatar`)
- Modifier: Block/element variant (e.g., `user-profile--premium`)
- NO Bootstrap classes in new components
- NO utility classes (except for one-off cases)

## Vue Best Practices
- MUST use Composition API with `<script setup>`
- MUST use TypeScript for all new components
- MUST add `data-qa` or `id` attributes to all interactive elements
- MUST use `key` with `v-for` directives
- MUST clean up timers in `onBeforeUnmount`
- NEVER use `v-html` without sanitization
- MUST use translation slugs (no hardcoded English)

## Component Structure
```vue
<template>
  <!-- Use data-qa for testing -->
  <button data-qa="submit-button">
    {{ t('submit.button') }}
  </button>
  
  <!-- Always use keys with v-for -->
  <div v-for="item in items" :key="item.id">
    {{ item.name }}
  </div>
</template>

<script setup lang="ts">
// Use TypeScript and Composition API
import { ref, onBeforeUnmount } from 'vue'
import type { Item } from '@/types'

// Define props with types
type Props = {
  items: Item[]
}

const props = defineProps<Props>()

// Clean up timers
const timer = ref()
onBeforeUnmount(() => {
  clearInterval(timer.value)
})

// Use camelCase for Amplitude properties
const trackEvent = () => {
  amplitudeV2('view_profile', {
    pageLocation: 'profile',
    userType: 'premium'
  })
}
</script>

<style>
/* Use BEM methodology */
.user-profile {
  &__avatar {
    /* styles */
  }
  
  &--premium {
    /* modifier styles */
  }
}
</style>
```

## Coding Guidelines

### Amplitude
- When passing a query parameter to be used for Amplitude's location parameter, ensure that the query parameter is named `location`.

### CSS
- Don't use Bootstrap classes in new components.
- If using more than two utility classes, those should be remade using custom classes.
- Check if typography classes can be remade into mixins so they can be reused within custom classes.
- **Use BEM (Block Element Modifier) methodology** for writing CSS classes.


### Writing Good Commit Messages
- Limit the subject line to 50 characters and wrap the body at 72 characters.
- Separate the subject from the body with a newline.
- Do not end the subject line with a period.
- Use the body to explain *what* and *why* as opposed to *how*.
- Use imperative mood in the subject line.

### Naming Conventions
#### File Naming
- Image files - `kebab-case`
- Vue component files - `PascalCase`
- All other project files - `camelCase`

#### Component Property Name
- Use `camelCase` when declaring:
  ```ts
  defineProps<{ greetingMessage: string }>()
  ```
- Use `kebab-case` when passing props to a child component:
  ```vue
  <MyComponent greeting-message="hello" />
  ```
- Follow the official Vue.js documentation approach.

#### Amplitude Property Name
- Use `camelCase` when passing additional properties to an Amplitude event:
  ```ts
  amplitudeV2(eventName, {
    propertyToPass: value
  });
  ```

### Vuex
- API calls meant to get information should have a `fetch` prefix.
- Getters should be named without a `get` prefix unless it's a function.
  ```ts
  getSomethingWithParam: () => (parameter) => '',
  something: () => ''
  ```

### API
- Each HTTP method should have its own prefix:
  ```ts
  get: {
    getUsersById: () => {},
    getSomething: () => {}
  },
  post: {
    postUser: () => {},
    postClientInformation: () => {},
    createUser: () => {},
    updateUser: () => {},
    createClientInformation: () => {},
    updateClientInformation: () => {}
  },
  patch: {
    patchUserInformation: () => {}
  },
  put: {
    putUserInformation: () => {}
  },
  delete: {
    deleteUserById: () => {}
  }
  ```
- Only `post` method can have variations like `post/update/create`.

### Basic Composables
- Avoid creating very basic composables (e.g., `useToggle`) if they don't bring significant value.

### Interfaces
- Prefix interface names with `I` to avoid name clashes with enums (e.g., `IProps`, `IAccount`).

### Button Attributes for E2E Tests
- All new `HButton` instances should use `data-qa`/`id` attributes to assist QA in constructing e2e tests.
- Avoid using auto-generated `v-qa-generate` tags unless necessary.

### Deprecations
- **Javascript files** → Use TypeScript instead. Acceptable to use `@ts-ignore` for initial refactor.
- **Vue Options API** → Use Vue Composition API with `<script setup>`.
- **Vuex** → Use Pinia instead.
- **Chargebee/non-Chargebee naming** → Legacy logic, avoid using.
- **Directive `v-trans` and component `Trans`** → Use `t()` function instead. Place inside `v-safe-html` for translated HTML content.
- **Interface keyword for Props or Emits** → Use `type` instead.
- **Avoid `index.ts` for re-exporting** due to circular dependencies.

### Translation Slugs
- All new files must use slugs for translations in the `hpanel` project.
- PRs with hardcoded English text instead of slugs should not be approved.
- **Ensure that all text strings are wrapped in the `t()` function for translation.**
  - For example, use `{{ t('v2.wordpress.installation.title') }}` instead of hardcoding 'WordPress Installation'.

### Testing
- All new `.vue` files must have corresponding tests created.
- **Tests should cover all props, events, and main functionality of the component.**

