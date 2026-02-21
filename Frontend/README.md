# Frontend Implementation

## Context
The backend (FastAPI at `localhost:8080`) is fully built with three endpoints:
- `POST /api/body-part` — selects body part, resets session
- `POST /api/diagnose` — uploads image, returns diagnosis + bounding box
- `POST /api/chat` — follow-up messages in conversation

No frontend exists yet. We need a React + TypeScript app at `Frontend/` with three pages that walk the user through: body part selection → image upload → diagnosis chat.

## Folder Structure

```
Frontend/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── context/
    │   └── AppContext.tsx
    ├── api/
    │   ├── bodyPart.ts
    │   ├── diagnose.ts
    │   └── chat.ts
    ├── types/
    │   └── index.ts
    ├── pages/
    │   ├── HomePage.tsx
    │   ├── UploadPage.tsx
    │   └── ChatPage.tsx
    ├── components/
    │   ├── BodySilhouette/
    │   │   ├── BodySilhouette.tsx
    │   │   └── BodySilhouette.module.css
    │   ├── ImageUploader/
    │   │   ├── ImageUploader.tsx
    │   │   └── ImageUploader.module.css
    │   ├── CameraCapture/
    │   │   ├── CameraCapture.tsx
    │   │   └── CameraCapture.module.css
    │   ├── DiagnosisImage/
    │   │   ├── DiagnosisImage.tsx
    │   │   └── DiagnosisImage.module.css
    │   ├── ChatWindow/
    │   │   ├── ChatWindow.tsx
    │   │   └── ChatWindow.module.css
    │   ├── ChatBubble/
    │   │   ├── ChatBubble.tsx
    │   │   └── ChatBubble.module.css
    │   ├── ChatInput/
    │   │   ├── ChatInput.tsx
    │   │   └── ChatInput.module.css
    │   └── LoadingSpinner/
    │       ├── LoadingSpinner.tsx
    │       └── LoadingSpinner.module.css
    └── styles/
        └── global.css
```

## Tech Stack
- **Vite 5** + **React 18** + **TypeScript 5**
- **React Router v6** for navigation
- **CSS Modules** for scoped component styles + `global.css` for design tokens
- **Fetch API** for all HTTP calls (no axios)
- No UI component library

## Dependencies (package.json)
```json
{
  "dependencies": { "react": "^18.3.1", "react-dom": "^18.3.1", "react-router-dom": "^6.26.0" },
  "devDependencies": {
    "@types/react": "^18.3.1", "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1", "typescript": "^5.5.4", "vite": "^5.4.1"
  }
}
```

## Routing
- `/` → HomePage
- `/upload` → UploadPage
- `/chat` → ChatPage
- `*` → redirect to `/`

## API Layer (`src/api/`)
- Uses Vite dev proxy: all `/api/*` requests proxied to `http://localhost:8080`
- `vite.config.ts` sets `server.proxy: { '/api': { target: 'http://localhost:8080', changeOrigin: true } }`
- Three thin fetch wrappers: `postBodyPart(id)`, `postDiagnose(file)`, `postChat(message)`

## State Management
- **AppContext** holds cross-page state: `selectedBodyPartId`, `selectedBodyPartName`, `uploadedImageFile`, `uploadedImageObjectUrl`, `diagnosis`, `boundingBox`
- `uploadedImageObjectUrl` is created via `URL.createObjectURL()` at upload time so ChatPage can show the image without re-reading the File
- Chat `messages[]` lives in local state on ChatPage only
- Navigation guards: UploadPage reads `useLocation().state?.bodyPartName`, redirects to `/` if missing; ChatPage reads context data, redirects to `/` if missing

## Page 1: HomePage
- **Banner at top**: full-width header with placeholder project title (`PROJECT NAME` as placeholder text), subtitle line (`AI-powered dermatology assistant` or similar placeholder)
- Renders `<BodySilhouette>` component below banner with a prompt like "Select the affected body area"
- On part click: calls `POST /api/body-part`, updates context, navigates to `/upload` with `state: { bodyPartName }`
- Shows `<LoadingSpinner>` and error message during/after API call

## BodySilhouette Component (SVG)
- Inline `<svg viewBox="0 0 200 500">` with shape elements for each body part:
  - `id=1` Face/Head: `<ellipse cx=100 cy=55 rx=35 ry=45>`
  - `id=2` Neck: `<rect x=85 y=98 width=30 height=28>`
  - `id=3` Chest/Torso: `<path d="M60 126 L140 126 L150 260 L50 260 Z">`
  - `id=5` Arm/Hand: two arm `<path>` shapes (left & right) — same id, clicking either sends id=5
  - `id=6` Leg/Foot: two leg `<path>` shapes — same id, clicking either sends id=6
  - `id=4` Back: torso automatically reports id=4 when body is rotated to back-facing
- Drag left/right to rotate: `scaleX(cos(rotY))` clamped to a minimum so it never collapses to a line
- Snaps to front (0°) or back (180°) on release with ease-out animation
- `hoveredPartId` local state drives CSS `.hovered` class (fill + stroke-width change)
- Tooltip div below SVG shows hovered part name

## Page 2: UploadPage
- `mode: 'file' | 'camera'` local state, toggled by two tab buttons
- File mode: `<ImageUploader>` with drag-and-drop, `accept="image/*"`, preview on selection
- Camera mode: `<CameraCapture>` using `navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })`; captures frame via hidden `<canvas>`, creates `File` from blob
- Submit button: calls `POST /api/diagnose` with `FormData`, stores result in context, navigates to `/chat` with `state: { fromUpload: true }`

## Page 3: ChatPage
- **Pinned top section**: `<DiagnosisImage imageObjectUrl boundingBox>` — image with red bounding box overlay
- **Diagnosis text**: initial LLM response rendered below image
- **Chat interface**: `<ChatWindow messages loading>` + `<ChatInput onSend disabled>`
- `messages` initialized as `[{ role: 'assistant', content: diagnosis }]`
- `handleSend()`: appends user message, calls `POST /api/chat`, appends assistant response
- "Start Over" button: calls `resetSession()` + navigates to `/`

## DiagnosisImage Bounding Box
- CSS absolute overlay technique: wrapper `div` is `position: relative`, bounding box is `position: absolute`
- Scale coordinates: `scaleX = img.clientWidth / img.naturalWidth`, same for Y
- `ResizeObserver` on wrapper recalculates on viewport resize

## Camera Capture
- `videoRef` + hidden `canvasRef` for frame capture
- `stream.getTracks().forEach(t => t.stop())` on capture and on unmount cleanup
- `URL.revokeObjectURL()` cleanup in `useEffect` return
- Stream assigned to video via `useEffect` watching `stream` state (avoids race condition)
- Stream stopped after `ctx.drawImage()` call, not before (avoids blank capture)

## Visual Design
**Color scheme (CSS variables in global.css):**
- `--color-bg-primary: #0a1628` (deep navy)
- `--color-bg-surface: #112240`
- `--color-bg-elevated: #1e3a5f`
- `--color-accent: #4a90d9`
- `--color-text-primary: #e8f0fe`
- `--color-bounding-box: #ff4444`

**Design principles (nice but simple):**
- Clean card-based layouts with subtle `box-shadow` and `border-radius: 12px` panels
- Smooth CSS transitions on hover states (0.15s ease)
- Consistent spacing scale from CSS variables
- `Inter` font via Google Fonts for a clean medical/tech feel
- Subtle gradient on the homepage banner: `linear-gradient(135deg, #112240 0%, #0a1628 100%)`
- All interactive elements (buttons, body part regions) have clear hover + focus states
- Page transitions feel instant and clean — no animation framework needed
- Shared `<Header>` component (app name banner) persists across all three pages with the placeholder title

## Build Order
1. Config files: `index.html`, `package.json`, `tsconfig.json`, `tsconfig.node.json`, `vite.config.ts`
2. `src/styles/global.css`, `src/types/index.ts`
3. API layer: `src/api/*.ts`
4. `src/context/AppContext.tsx`
5. `src/main.tsx`, `src/App.tsx`
6. `LoadingSpinner` component + `Header` component (shared banner)
7. `BodySilhouette` component → `HomePage.tsx` (end-to-end Page 1)
8. `ImageUploader` + `CameraCapture` components → `UploadPage.tsx` (end-to-end Page 2)
9. `DiagnosisImage`, `ChatBubble`, `ChatWindow`, `ChatInput` → `ChatPage.tsx` (end-to-end Page 3)

**`Header` component** (`src/components/Header/Header.tsx`): Shared banner rendered at top of all three pages. Contains placeholder `PROJECT NAME` title and a subtle subtitle. Styled with the gradient background. Exported and imported in each page component.

## Verification
1. `cd Frontend && npm install && npm run dev`
2. Ensure backend is running: `cd Backend && uvicorn main:app --reload --port 8080`
3. Navigate to `http://localhost:3000`
4. Page 1: Hover body parts (should highlight), click one (should navigate to /upload)
5. Page 2: Upload an image file, click Submit (should navigate to /chat with spinner during processing)
6. Page 3: Verify image displays with red bounding box, initial diagnosis text loads, send a follow-up message and verify response appears
7. Test camera capture: switch to camera mode on Page 2, capture a photo, submit
8. Test "Start Over" on Page 3 returns to home and clears state
9. Run `npm run typecheck` — should have zero TypeScript errors
