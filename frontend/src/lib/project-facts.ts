// Presentation constants verified against docs/final_experimental_results.md,
// lines 12–15 and 124–140 at research release 2854d153d65c93689875884f3ad45f0b3b468ec2.
// These describe the frozen research graph, never a connected model/session.
export const projectFacts = {
  dataset: 'CSE-CIC-IDS2018',
  model: 'HTGNN',
  inferenceMode: 'Attention-pruned',
  inputFeatures: 78,
  retainedFeatures: 61,
  removedFeatures: 17,
  // Original, zero-based IDs. Documented frozen policy; no feature names inferred.
  removedFeatureIds: [6, 10, 29, 32, 34, 35, 40, 41, 42, 50, 52, 57, 64, 70, 71, 72, 73],
  categoryNodes: 8,
  unprunedNodes: 87,
  prunedNodes: 70,
  unprunedEdges: 86,
  prunedEdges: 69,
  windowSize: 5,
} as const
