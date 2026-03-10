// ─── Grade options ────────────────────────────────────────────────────────────

export const VGRADE_OPTIONS = [
  "V0-", "V0", "V0+", "V1", "V1+", "V2", "V2+", "V3", "V3+",
  "V4", "V4+", "V5", "V5+", "V6", "V6+", "V7", "V7+", "V8", "V8+",
  "V9", "V9+", "V10", "V10+", "V11", "V11+", "V12", "V12+",
  "V13", "V13+", "V14", "V14+", "V15", "V15+", "V16",
];

export const FONT_OPTIONS = [
  "4a", "4b", "4c", "5a", "5b", "5c",
  "6a", "6a+", "6b", "6b+", "6c", "6c+",
  "7a", "7a+", "7b", "7b+", "7c", "7c+",
  "8a", "8a+", "8b", "8b+", "8c", "8c+",
];

// ─── Name generator ──────────────────────────────────────────────────────────

const ADJECTIVES = [
  "Angry", "Bold", "Cosmic", "Dancing", "Electric", "Fierce", "Gnarly",
  "Humble", "Icy", "Jazzy", "Killer", "Lunar", "Mighty", "Noble",
  "Obscure", "Primal", "Quantum", "Radical", "Savage", "Twisted",
  "Ultra", "Voluptuous", "Wicked", "Xtreme", "Yonder", "Zesty",
];

const ANIMALS = [
  "Aardvark", "Badger", "Cobra", "Dolphin", "Eagle", "Falcon", "Gorilla",
  "Hawk", "Ibex", "Jaguar", "Koala", "Lemur", "Mantis", "Narwhal",
  "Osprey", "Panther", "Quokka", "Raven", "Scorpion", "Tiger",
  "Urchin", "Viper", "Wolf", "Xingu River Ray", "Yak", "Zebra",
];

export function generateClimbName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const animal = ANIMALS[Math.floor(Math.random() * ANIMALS.length)];
  return `${adj} ${animal}`;
}
