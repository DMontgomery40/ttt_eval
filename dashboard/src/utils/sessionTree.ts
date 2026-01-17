import type { SessionIndex, SessionTreeNode } from '../types';

// Build session tree from index (git-like lineage)
export function buildSessionTree(index: SessionIndex): SessionTreeNode[] {
  const roots: SessionTreeNode[] = [];
  const nodeMap = new Map<string, SessionTreeNode>();

  for (const session of Object.values(index.sessions)) {
    nodeMap.set(session.session_id, {
      session,
      children: [],
      depth: 0,
    });
  }

  for (const session of Object.values(index.sessions)) {
    const node = nodeMap.get(session.session_id);
    if (!node) continue;

    if (session.parent_session_id === null) {
      roots.push(node);
    } else {
      const parent = nodeMap.get(session.parent_session_id);
      if (parent) parent.children.push(node);
      else roots.push(node); // orphan, treat as root
    }
  }

  function setDepths(node: SessionTreeNode, depth: number) {
    node.depth = depth;
    for (const child of node.children) setDepths(child, depth + 1);
  }

  for (const root of roots) setDepths(root, 0);

  return roots;
}

// Get lineage path from root to session
export function getSessionLineage(sessionId: string, index: SessionIndex): string[] {
  const lineage: string[] = [];
  let current = index.sessions[sessionId];

  while (current) {
    lineage.unshift(current.session_id);
    if (current.parent_session_id) current = index.sessions[current.parent_session_id];
    else break;
  }

  return lineage;
}

