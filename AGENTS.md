## File Size

- Target file size is roughly 300-500 lines.
- If a file is clearly growing past that range, prefer splitting it.
- Split by cohesive responsibility, not arbitrarily.
- Small exceptions are fine when the language or framework strongly favors a
  single file, or when a split would make navigation worse.

## Code Organization

- Group code by functionality.
- Keep related functions near each other.
- Prefer files that have one clear job.
- Prefer mostly flat module structure.
- Avoid deep module nesting unless there is a strong, concrete reason for it.
- Small groups are fine where they clearly improve ownership or navigation.
- Prefer modules organized around behavior and ownership, not around vague
  categories like `utils` unless the helpers are truly shared and generic.
- In backend code, keep request parsing, validation, business logic, and data
  access reasonably separated unless the code is trivial.

## Style

- Default style target is "C+" / "C+-" code:
  - direct
  - explicit
  - readable
  - low-magic
  - modest abstraction
  - easy to trace in a debugger
- Prefer straightforward control flow over clever indirection.
- Avoid deep nesting in `if` statements, `match` expressions, and general logic.
- Prefer early returns, helper functions, and flatter branching when that keeps
  behavior clearer.
- Prefer obvious data movement over framework tricks.
- Prefer small helper functions over deep abstraction stacks.
- Avoid unnecessary genericization.
- Avoid premature reuse that makes local behavior harder to understand.
- Keep naming concrete and descriptive.

## Language Pragmatism

- Do not force C-like structure where it clearly fights the language.
- Follow the grain of the language when the idiomatic form is materially better.
- In Rust, this means using enums, pattern matching, `Result`, and normal
  ownership-based design instead of awkward pseudo-C emulation.
- The rule is: prefer explicitness first, but do not write unnatural code just
  to mimic C.

## Configuration Philosophy

- Hardcoded connection details are acceptable in this private workspace when
  that is the established local convention.
- This is an intentional tradeoff to reduce bugs caused by mismatched env vars,
  stale env files, deployment drift, and config being partially copied over chat.
- Do not introduce env-var churn by default.
- Do not "clean up" hardcoded private connection details unless explicitly asked.
- If a platform or deployment target requires env vars, use them there, but do
  not automatically push the whole codebase toward env-driven config.
- Prefer one obvious source of truth over multiple partially overlapping config
  paths.

## Change Discipline

- Preserve existing behavior unless the task is to change it.
- Do not perform broad style rewrites unless requested.
- Do not move code across many files just to satisfy a size guideline if the
  current structure is still easier to work with.
- When splitting files, keep the resulting layout obvious and boring.

## Backend Bias

- Keep SQL and DB-facing behavior easy to inspect.
- Favor explicit queries and explicit mapping over hidden ORM magic.
- Keep schema-affecting behavior close to the code that depends on it.
- Avoid config patterns that make it unclear which database or service is being
  targeted.

## When in Doubt

- Choose the more explicit option.
- Choose the easier-to-debug option.
- Choose the layout that keeps related functionality together.
- Choose the solution with fewer hidden config dependencies.