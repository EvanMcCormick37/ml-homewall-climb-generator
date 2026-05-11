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
