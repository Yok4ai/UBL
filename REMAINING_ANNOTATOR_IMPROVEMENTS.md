# Remaining Annotator Improvements

## Completed ✅
1. **Label merging with dataset.yaml** - Backend now reads existing classes and merges
2. **Persist state across tab switches** - localStorage saves boxes/classNames/scale
3. **Improved resize handles** - Photoshop-style: larger (10px), white fill with orange border, 8 handles (corners + midpoints)

## Still Need to Add

### 1. Confidence Threshold Slider
Add to toolbar (after zoom controls):
```tsx
{/* Confidence Threshold Controls */}
<div className="flex items-center gap-2 border-r border-[#3a3a3a] pr-4">
  <span className="text-sm text-gray-400">Confidence:</span>
  <input
    type="range"
    min="0.05"
    max="0.5"
    step="0.05"
    value={confidenceThreshold}
    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
    className="w-24"
  />
  <span className="text-sm text-gray-300 w-12">{(confidenceThreshold * 100).toFixed(0)}%</span>
</div>
```

Update `commitEditingBox` to use these thresholds instead of hard-coded 0.18.

### 2. Hand/Selection Tool Switcher
Add to toolbar:
```tsx
{/* Tool Switcher */}
<div className="flex items-center gap-1 border-r border-[#3a3a3a] pr-4">
  <button
    onClick={() => setTool('select')}
    className={`px-3 py-2 rounded ${tool === 'select' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
    title="Selection Tool (V)"
  >
    ✋ Select
  </button>
  <button
    onClick={() => setTool('hand')}
    className={`px-3 py-2 rounded ${tool === 'hand' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300'}`}
    title="Hand Tool (H)"
  >
    ✋ Pan
  </button>
</div>
```

Update mouse handlers to check tool state - if `tool === 'hand'`, enable panning instead of drawing.

### 3. Make Labels Interactive (Quick Add)
Update label chips to be clickable:
```tsx
{classNames.map((className) => (
  <button
    key={className}
    onClick={() => {
      setPredictLabels(className);
      handlePredict();
    }}
    className="flex items-center gap-2 px-3 py-1 rounded text-sm text-white shadow-md hover:opacity-80 transition-opacity cursor-pointer"
    style={{ backgroundColor: getColorForLabel(className) }}
    title={`Click to auto-detect all "${className}" objects`}
  >
    {className}
    <span className="text-xs opacity-75">
      ({boxes.filter(b => b.label === className).length})
    </span>
    <span className="text-xs">+</span>
  </button>
))}
```

### 4. Pan Functionality
In `handleCanvasMouseDown`:
```tsx
if (tool === 'hand') {
  setIsPanning(true);
  setPanStart({ x: e.clientX, y: e.clientY });
  return;
}
```

In `handleCanvasMouseMove`:
```tsx
if (isPanning && panStart && containerRef.current) {
  const dx = e.clientX - panStart.x;
  const dy = e.clientY - panStart.y;
  containerRef.current.scrollLeft -= dx;
  containerRef.current.scrollTop -= dy;
  setPanStart({ x: e.clientX, y: e.clientY });
  return;
}
```

In `handleCanvasMouseUp`:
```tsx
if (isPanning) {
  setIsPanning(false);
  setPanStart(null);
  return;
}
```

### 5. Keyboard Shortcut for Tools
Add to keyboard handler:
```tsx
if (e.key === 'v' || e.key === 'V') {
  setTool('select');
}
if (e.key === 'h' || e.key === 'H') {
  setTool('hand');
}
```

### 6. Update Cursor Based on Tool
Update canvas className:
```tsx
className={`border border-[#2a2a2a] shadow-2xl rounded ${tool === 'hand' ? 'cursor-grab' : 'cursor-crosshair'} ${isPanning ? 'cursor-grabbing' : ''}`}
```

## Testing Checklist
- [ ] Confidence slider changes detection sensitivity
- [ ] Hand tool allows panning zoomed image
- [ ] V/H keys switch tools
- [ ] State persists across tab switches
- [ ] Labels merge with existing dataset.yaml (vim_liquid becomes class 62, not class 0)
- [ ] Clicking label chip auto-detects that class
- [ ] Resize handles are larger and easier to grab
- [ ] Mid-point handles work for resizing
