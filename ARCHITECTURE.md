# Comprehensive Project Plan: BetaZero Climb Generator

## Executive Summary

## After analyzing your codebase and project outline, I've identified several critical design decisions that must be resolved before proceeding, along with gaps and inefficiencies in the current plan. This document provides a restructured, actionable roadmap.

## Part 1: Critical Design Decisions (Resolve Before Coding)

These decisions have cascading effects on the entire architecture. Making the wrong choice here means refactoring later.

### Decision 1: Climb Representation for Diffusion

The Problem: Your current Holdset structure is:

```
interface Holdset {
  start: number[];   // hold indices
  finish: number[];
  hand: number[];
  foot: number[];
}
```

This is a _sparse categorical_ representation (which holds are used, and their categories). Diffusion models work best in _continuous dense_ spaces.

**Options:**

| Option                                 | Description                                                                                              | Pros                                                          | Cons                                                                                |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **A: Binary mask + category heads**    | Model predicts P(hold_used) for each hold, then separate heads for P(start\|used), P(finish\|used), etc. | Clean separation of concerns; works with variable hold counts | Two-stage inference; category predictions may be inconsistent                       |
| **B: Multi-class per hold**            | Model predicts 5-class distribution per hold: {unused, start, finish, hand, foot}                        | Single forward pass; consistent                               | Class imbalance (most holds unused); doesn't naturally handle "both start AND hand" |
| **C: Multi-label per hold**            | Model predicts independent P(start), P(finish), P(hand), P(foot) per hold                                | Handles overlapping categories                                | More outputs; need to handle correlations                                           |
| **D: Continuous embedding + decoding** | Encode entire climb as fixed-dim vector, diffuse in that space, decode to holds                          | Most "pure" diffusion approach                                | Requires encoder-decoder architecture; reconstruction loss may be lossy             |

**My Recommendation:** Option A or C, depending on whether holds can belong to multiple categories (can a start hold also be a hand hold?). Looking at your current UI, you cycle through categories, implying mutual exclusivity → **Option B is cleanest**.

**Action Required:** Decide if categories are mutually exclusive. If yes, use Option B. If a hold can be both "start" and "hand", use Option C.

---

### Decision 2: Conditioning Mechanism

**The Problem:** Users want to generate climbs by specifying:

- Target grade (V0-V17)
- Wall angle
- Style tags (crimpy, dynamic, etc.)
- Specific holds to include/exclude

**Options:**

| Option                                  | Description                                             | Complexity                      |
| --------------------------------------- | ------------------------------------------------------- | ------------------------------- |
| **Classifier-Free Guidance**            | Train with random condition dropout, guide at inference | Medium - proven approach        |
| **Conditional embedding concatenation** | Embed conditions, concat to input                       | Low - simpler but less flexible |
| **Cross-attention conditioning**        | Conditions attend to hold features                      | High - most expressive          |

**My Recommendation:** Start with **conditional embedding concatenation** for simplicity. Grade/angle as continuous, tags as multi-hot encoded, required/excluded holds as binary masks per hold.

---

### Decision 3: Model Architecture Choice

**The Problem:** "Diffusion model" is broad. What specifically?

**Options:**

| Architecture                  | Best For                 | Inference Speed   | Implementation Complexity |
| ----------------------------- | ------------------------ | ----------------- | ------------------------- |
| **DDPM (standard)**           | High quality             | Slow (many steps) | Medium                    |
| **DDIM**                      | Quality + speed tradeoff | Medium            | Medium                    |
| **Flow Matching**             | Fast inference           | Fast              | Higher                    |
| **Discrete Diffusion (D3PM)** | Categorical data         | Medium            | Higher                    |

**My Recommendation:** Given your data is inherently discrete (which holds are selected), consider **D3PM (Discrete Denoising Diffusion)** or **standard DDPM with continuous relaxation** (treat categories as probabilities during diffusion, discretize at output).

**Action Required:** Prototype both approaches on a small dataset before committing.

---

### Decision 4: Aurora Data Structure & Legal

**Critical Questions:**

1. What format is Aurora data in? (API? Database dump? Scraped?)
2. Do you have rights to redistribute/use this data commercially?
3. How do Aurora hold coordinates map to your schema?
4. Do Aurora boards have fixed layouts (same holds always in same positions)?

**Action Required:** Before any data work:

- Document the Aurora data format you have access to
- Verify licensing/terms of use
- Create a mapping document: Aurora schema → BetaZero schema

---

### Decision 5: Generation UX - Synchronous vs Asynchronous

**The Problem:** Diffusion models can be slow. How does the UX handle this?

| Approach                     | UX                               | Backend Complexity               |
| ---------------------------- | -------------------------------- | -------------------------------- |
| **Synchronous**              | User waits 2-10s, sees result    | Simple - single request          |
| **Asynchronous (polling)**   | User submits, polls for result   | Medium - needs job queue         |
| **Asynchronous (WebSocket)** | User submits, gets pushed result | Higher - needs WS infrastructure |
| **Optimistic + refinement**  | Show fast rough result, refine   | Highest - two-stage model        |

**My Recommendation:** Start with **synchronous** if inference < 5s. Fall back to **async polling** if needed. Your existing `jobs` types suggest you planned for async.

**Action Required:** Benchmark your model's inference time before deciding.

---

### Decision 6: Sharing Mechanism

**The Problem:** "Users should be able to easily share generated climbs with others, possibly as a link."

**Options:**

| Approach                    | URL Example                                  | Storage                   | Privacy               |
| --------------------------- | -------------------------------------------- | ------------------------- | --------------------- |
| **Persist all generations** | `/climbs/abc123`                             | Every generation saved    | Need cleanup policy   |
| **Encode in URL**           | `/generate?params=base64...`                 | Stateless                 | Limited by URL length |
| **Share = explicit save**   | Generate → "Save & Share" → `/climbs/abc123` | Only saved climbs persist | Clear user intent     |

**My Recommendation:** **Option 3** - Generated climbs are ephemeral until user explicitly saves. Saved climbs get a shareable URL. This avoids database bloat and makes user intent clear.

---

### Decision 7: Authentication Model

**Current mentions:**

- Setter password for editing holds
- Private walls
- Public sharing

**Proposed coherent model:**

```
Wall:
  - owner_password_hash (for editing)
  - visibility: "public" | "unlisted" | "private"
    - public: appears in list, anyone can view
    - unlisted: not in list, anyone with link can view
    - private: not in list, requires password to view

Climb:
  - wall_id (inherits wall visibility by default)
  - is_generated: boolean
  - visibility_override: null | "public" | "unlisted"
```

**Action Required:** Finalize auth model before implementing any password/privacy features.

---

## Part 2: Restructured Project Plan

Based on the decisions above, here's a revised, ordered plan with dependencies made explicit.

### Phase 0: Foundation & Decisions (Week 1)

**0.1 Finalize Design Decisions**

- [ ] Document choice for each of the 7 decisions above
- [ ] Create `ARCHITECTURE.md` with chosen approaches
- [ ] Review with any collaborators/stakeholders

**0.2 Audit Current Codebase**

- [ ] Document current backend API (you haven't shared it - I'm inferring from frontend)
- [ ] List all existing endpoints and their implementations
- [ ] Identify what's implemented vs stubbed
- [ ] Create `CURRENT_STATE.md`

**0.3 Define Data Contracts**

- [ ] Finalize `HoldDetail` schema (is current sufficient for ML?)
- [ ] Finalize `Climb` schema (add `is_generated`, `source` fields?)
- [ ] Define `GenerationRequest` schema
- [ ] Define `GenerationResult` schema
- [ ] Version these as OpenAPI spec

---

### Phase 1: Data Pipeline (Weeks 2-3)

**1.1 Aurora Data Acquisition**

- [ ] Document Aurora data source and format
- [ ] Verify legal right to use data
- [ ] Write extraction script (if needed)
- [ ] Store raw data in version-controlled location

**1.2 Aurora Data Transformation**

- [ ] Map Aurora board layouts to `WallMetadata` schema
  - [ ] Handle different board sizes (Kilter 8x10, 12x12, etc.)
  - [ ] Map Aurora hold IDs to your hold indexing
  - [ ] Extract hold positions (x, y in feet)
  - [ ] Handle LED positions → hold positions mapping
- [ ] Map Aurora climbs to `Climb` schema
  - [ ] Map Aurora grades to your V-scale (0-170)
  - [ ] Map Aurora angles to your angle field
  - [ ] Transform hold lists to `Holdset` structure
  - [ ] Filter to climbs with ≥1 ascent (as you mentioned)
- [ ] Write transformation scripts with validation
- [ ] Create data quality report

**1.3 Data Loading**

- [ ] Create bulk upload endpoint: `POST /walls/bulk`
- [ ] Create bulk climb upload: `POST /walls/{id}/climbs/bulk`
- [ ] Load all Aurora walls
- [ ] Load all Aurora climbs
- [ ] Verify data integrity in database

**1.4 ML Dataset Preparation**

- [ ] Export training data: (wall_id, hold_features, climb_holdset, grade, angle)
- [ ] Create train/validation/test splits (by wall? by time? random?)
- [ ] Compute dataset statistics (grade distribution, holds per climb, etc.)
- [ ] Save as efficient format (Parquet, TFRecord, or PyTorch datasets)

---

### Phase 2: Model Development (Weeks 4-7)

**2.1 Baseline Model**

- [ ] Implement simple baseline (random hold selection)
- [ ] Define evaluation metrics:
  - [ ] Reconstruction accuracy (on held-out climbs)
  - [ ] Grade prediction accuracy (train classifier, check generated climbs)
  - [ ] Validity rate (start + finish present, reasonable # holds)
  - [ ] Diversity (unique climbs generated)
- [ ] Establish baseline scores

**2.2 Feature Engineering**

- [ ] Design hold embedding: how to represent a single hold
  - Position (x, y) - normalized to wall dimensions
  - Physical features (pull_x, pull_y, useability, is_foot)
  - Contextual features (distance to neighbors, local density?)
- [ ] Design wall embedding: how to represent entire wall
  - Set of hold embeddings
  - Wall metadata (angle, dimensions)
- [ ] Design condition embedding
  - Grade (scalar, normalized)
  - Angle (scalar, normalized)
  - Tags (multi-hot or learned embeddings)

**2.3 Diffusion Model Implementation**

- [ ] Choose framework (PyTorch recommended)
- [ ] Implement noise schedule
- [ ] Implement forward diffusion process
- [ ] Implement denoising network
  - Architecture choice: Transformer? Graph neural network? MLP?
  - I recommend **Set Transformer** or **Perceiver** for variable-size hold sets
- [ ] Implement conditional embedding injection
- [ ] Implement loss function
- [ ] Write training loop with logging (Weights & Biases recommended)

**2.4 Training Infrastructure**

- [ ] Set up GPU training environment (local or cloud)
- [ ] Implement checkpointing
- [ ] Implement early stopping
- [ ] Create training config system (Hydra or simple YAML)

**2.5 Training & Iteration**

- [ ] Train initial model
- [ ] Evaluate on test set
- [ ] Analyze failure cases
- [ ] Iterate on architecture/hyperparameters
- [ ] Document final model choice with rationale

**2.6 ClimbGenerator Wrapper**

- [ ] Implement `ClimbGenerator` class that wraps diffusion model
- [ ] Add inference-time features:
  - [ ] Classifier-free guidance for grade control
  - [ ] Temperature/diversity control
  - [ ] Required/excluded hold constraints
  - [ ] Post-processing (ensure valid start/finish)
- [ ] Optimize inference speed
  - [ ] DDIM for fewer steps
  - [ ] Batching for multiple generations
  - [ ] ONNX export if needed
- [ ] Write comprehensive unit tests

---

### Phase 3: Backend Integration (Weeks 8-9)

**3.1 Model Serving Setup**

- [ ] Decide serving approach:
  - [ ] In-process (model loaded in FastAPI)
  - [ ] Separate service (gRPC or REST)
  - [ ] Managed service (SageMaker, Vertex AI)
- [ ] Implement model loading with versioning
- [ ] Add health check endpoint for model readiness

**3.2 Generation API**

- [ ] Implement `POST /walls/{wall_id}/generate` endpoint

```
  Request:
    grade: int | null
    angle: int | null
    tags: string[] | null
    required_holds: int[] | null
    excluded_holds: int[] | null
    count: int (default 1, max 10)

  Response:
    generations: [{
      holdset: Holdset,
      predicted_grade: int,
      confidence: float
    }]
```

- [ ] Add request validation
- [ ] Add rate limiting
- [ ] Add generation logging for analytics

**3.3 Save Generated Climb**

- [ ] Extend `POST /walls/{wall_id}/climbs` to accept `source: "generated"`
- [ ] Store generation metadata (model version, input params)

**3.4 Async Generation (if needed)**

- [ ] Set up job queue (Redis + RQ, or Celery)
- [ ] Implement `POST /walls/{wall_id}/generate/async` → returns job_id
- [ ] Implement `GET /jobs/{job_id}` for polling
- [ ] Implement job cleanup for old jobs

---

### Phase 4: Frontend - Generation Feature (Week 10)

**4.1 Generation Page UI**

- [ ] Implement `/walls/$wallId/generate` route (currently stubbed)
- [ ] Design generation form:
  - Grade selector (slider or dropdown)
  - Angle selector (if wall supports multiple)
  - Tag multi-select
  - "Include holds" mode (click to require)
  - "Exclude holds" mode (click to exclude)
- [ ] Design results display:
  - Show generated climb on wall canvas
  - Show predicted grade/confidence
  - "Generate Another" button
  - "Save This Climb" button
  - "Share" button

**4.2 Generation Flow**

- [ ] Implement API call to generation endpoint
- [ ] Add loading state with appropriate feedback
- [ ] Handle errors gracefully
- [ ] Implement "Save" flow (transitions to view page)

**4.3 Sharing Feature**

- [ ] Implement share URL generation for saved climbs
- [ ] Create `/climbs/{climbId}` public view page (no wall context needed?)
- [ ] Or: `/walls/{wallId}/climbs/{climbId}` with wall context
- [ ] Add copy-to-clipboard for share URL
- [ ] Add social sharing buttons (optional)

**4.4 Aurora Wall Selection**

- [ ] Update home page wall selector to show Aurora walls prominently
- [ ] Add wall categorization (user walls vs Aurora walls)
- [ ] Consider wall preview thumbnails

---

### Phase 5: Enhanced Hold Editor (Week 11)

**5.1 Bounding Quadrilateral for Coordinates**

- [ ] Design UI for quadrilateral selection
  - Four corner drag handles
  - Labels for top/bottom/left/right edges
  - Preview of coordinate grid
- [ ] Implement perspective transform math

```
  Input: 4 corners in pixel space, wall dimensions in feet
  Output: Function mapping pixel → feet coordinates
```

- [ ] Store quadrilateral corners in wall metadata
- [ ] Update hold coordinate calculation to use transform

**5.2 Kickboard Support**

- [ ] Add second quadrilateral for kickboard region
- [ ] UI to toggle kickboard mode
- [ ] Store kickboard holds separately or with flag
- [ ] Handle kickboard in climb generation (some climbs use kickboard, some don't)

**5.3 Migration**

- [ ] Add migration for existing walls (default quadrilateral = image bounds)
- [ ] Tool to re-calculate hold coordinates for existing walls

---

### Phase 6: Access Control (Week 12)

**6.1 Wall Password Protection**

- [ ] Add `setter_password_hash` to Wall schema
- [ ] On wall creation, optionally set password
- [ ] Protect `PUT /walls/{id}/holds` with password
- [ ] Protect `DELETE /walls/{id}` with password
- [ ] UI for entering password when editing

**6.2 Wall Visibility**

- [ ] Add `visibility` enum to Wall schema
- [ ] Update `GET /walls` to filter by visibility
- [ ] Add `GET /walls/{id}` logic:
  - Public/unlisted: return wall
  - Private: require password or return 404
- [ ] UI for setting visibility on creation/edit
- [ ] "Unlisted" walls accessible by direct link

**6.3 Password Security**

- [ ] Use bcrypt for password hashing
- [ ] Implement rate limiting on password attempts
- [ ] Never return password hash in API responses

---

### Phase 7: Cleanup & Polish (Week 13)

**7.1 Bug Fixes**

- [ ] Systematic testing of all features
- [ ] Fix identified bugs
- [ ] Handle edge cases (empty walls, no climbs, etc.)

**7.2 UI Polish**

- [ ] Consistent design system (colors, spacing, typography)
- [ ] Responsive design (mobile support)
- [ ] Loading states everywhere
- [ ] Error states everywhere
- [ ] Empty states everywhere
- [ ] Accessibility audit (keyboard nav, screen readers)

**7.3 Performance**

- [ ] Frontend bundle analysis and optimization
- [ ] Backend query optimization
- [ ] Add caching where appropriate
- [ ] Lazy loading for large lists

**7.4 Documentation**

- [ ] API documentation (auto-generated from OpenAPI)
- [ ] Code comments on complex logic
- [ ] README with setup instructions
- [ ] CONTRIBUTING guide
- [ ] Architecture documentation

---

### Phase 8: Deployment (Week 14)

**8.1 Backend Containerization**

- [ ] Write `Dockerfile` for backend
  - Multi-stage build for smaller image
  - Include ML model in image or load from storage?
- [ ] Write `docker-compose.yml` for local development
- [ ] Test container locally
- [ ] Push to DockerHub

**8.2 Database Setup**

- [ ] Choose managed database (Cloud SQL, RDS, or self-hosted)
- [ ] Set up production database
- [ ] Configure backups
- [ ] Run migrations

**8.3 Backend Hosting**

- [ ] Set up GCP VM (or Cloud Run for easier scaling)
- [ ] Configure networking (firewall, SSL)
- [ ] Set up reverse proxy (nginx or Cloud Load Balancer)
- [ ] Deploy container
- [ ] Set up monitoring (Cloud Monitoring or self-hosted)
- [ ] Set up logging (Cloud Logging)
- [ ] Set up alerts

**8.4 Frontend Deployment**

- [ ] Configure Vercel project
- [ ] Set up environment variables (API URL)
- [ ] Deploy to Vercel
- [ ] Test production build

**8.5 Domain Setup**

- [ ] Purchase domain
- [ ] Configure DNS for frontend (Vercel)
- [ ] Configure DNS for API subdomain (api.yourdomain.com)
- [ ] Ensure HTTPS everywhere
- [ ] Test end-to-end

---

### Phase 9: Launch Content (Week 15)

**9.1 Dev Blog Post**

- [ ] Outline: problem statement, architecture overview, key decisions
- [ ] Write draft
- [ ] Create architecture diagrams
- [ ] Add code snippets for interesting parts
- [ ] Review and edit
- [ ] Publish on Substack

**9.2 ML Blog Post**

- [ ] Outline: why diffusion, data representation, model architecture, results
- [ ] Write draft
- [ ] Include training curves, sample outputs
- [ ] Compare to baselines
- [ ] Discuss limitations and future work
- [ ] Review and edit
- [ ] Publish on Substack

**9.3 Demo Content**

- [ ] Create demo video showing generation flow
- [ ] Prepare sample generated climbs for each Aurora board
- [ ] Screenshot gallery for social media

---

### Phase 10: Launch & Promotion (Week 16)

**10.1 Soft Launch**

- [ ] Share with climbing friends for feedback
- [ ] Fix critical issues found
- [ ] Gather testimonials

**10.2 Public Launch**

- [ ] Post on Reddit (r/climbing, r/climbharder, r/homewalls)
- [ ] Post on climbing forums
- [ ] Share on Twitter/X
- [ ] Post on Hacker News (if technical angle is compelling)
- [ ] Share Substack posts

**10.3 Ongoing**

- [ ] Monitor for issues
- [ ] Respond to user feedback
- [ ] Track analytics
- [ ] Plan v2 features based on feedback

---

## Part 3: Risk Analysis & Mitigations

| Risk                                   | Impact                 | Likelihood | Mitigation                                                                               |
| -------------------------------------- | ---------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| **Model doesn't generate good climbs** | High - core value prop | Medium     | Start with simple baseline, iterate. Have fallback to curated climbs.                    |
| **Aurora data issues (format/legal)**  | High - blocks Phase 1  | Low-Medium | Verify data access early. Have backup plan (synthetic data, user-submitted climbs only). |
| **Inference too slow**                 | Medium - bad UX        | Medium     | Optimize early. Use async if needed. Consider edge deployment.                           |
| **Nobody uses it**                     | Medium - motivation    | Medium     | Launch MVP fast, get feedback early. Don't over-build before validation.                 |
| **Scope creep**                        | Medium - delays        | High       | Strict prioritization. Mark features as "v1" vs "v2".                                    |
| **Security issues**                    | High - reputation      | Low        | Follow security best practices. No real authentication = limited risk.                   |

---

## Part 4: Questions I Need Answered

Before you proceed, please clarify:

1. **Backend stack**: What's your current backend? FastAPI? Django? Node?

2. **Database**: What database are you using? PostgreSQL? MongoDB?

3. **Aurora data**: Do you have this data already? What format?

4. **ML experience**: What's your comfort level with PyTorch, diffusion models?

5. **Timeline**: Is the 16-week estimate realistic? Any hard deadlines?

6. **Resources**: Do you have GPU access for training? Budget for cloud hosting?

7. **Category exclusivity**: Can a hold be both "start" AND "hand"? Or are categories mutually exclusive?

8. **Existing models**: You mentioned `jobs` and `models` - is there existing generation code to review?

---

## Appendix: Suggested File Structure

```
betazero/
├── climb-api/                    # Backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── walls.py
│   │   │   │   ├── climbs.py
│   │   │   │   ├── generation.py
│   │   │   │   └── jobs.py
│   │   │   └── deps.py
│   │   ├── models/              # DB models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── services/
│   │   │   ├── generation/
│   │   │   │   ├── generator.py
│   │   │   │   └── model_loader.py
│   │   │   └── ...
│   │   └── main.py
│   ├── ml/                       # ML code
│   │   ├── data/
│   │   │   ├── aurora_loader.py
│   │   │   └── dataset.py
│   │   ├── models/
│   │   │   ├── diffusion.py
│   │   │   └── embeddings.py
│   │   ├── training/
│   │   │   ├── train.py
│   │   │   └── evaluate.py
│   │   └── configs/
│   ├── Dockerfile
│   └── requirements.txt
├── climb-frontend/               # Frontend (your current code)
├── data/
│   ├── raw/                     # Raw Aurora data
│   ├── processed/               # Transformed data
│   └── models/                  # Trained model checkpoints
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── DEPLOYMENT.md
└── docker-compose.yml
```
