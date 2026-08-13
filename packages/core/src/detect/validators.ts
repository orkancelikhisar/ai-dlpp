export type Validator = (candidate: string) => boolean;

const REGISTRY: Record<string, Validator> = {
  "pan-structure": () => true, // implemented in Task 5
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
