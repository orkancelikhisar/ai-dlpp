export type Validator = (candidate: string) => boolean;

// Fail closed until Task 5 lands the real implementations: a stub that returned true
// would silently validate nothing. The loader only checks that names resolve, so a
// registered-but-unimplemented validator is safe to declare and unsafe to call.
const REGISTRY: Record<string, Validator> = {
  "pan-structure": () => {
    throw new Error("pan-structure not implemented (Task 5)");
  },
};

export function hasValidator(name: string): boolean {
  // Own keys only: `in` would consult the prototype chain, so a rule naming
  // "toString" would pass the loader's check and then fail at getValidator time.
  return Object.hasOwn(REGISTRY, name);
}

export function getValidator(name: string): Validator {
  const v = REGISTRY[name];
  if (!v) throw new Error(`unknown validator "${name}"`);
  return v;
}
