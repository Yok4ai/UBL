'use client';

import { useState, useEffect, useRef } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Save,
  Trash2,
  Hand,
  MousePointer2,
  Sparkles,
  ChevronDown,
  ChevronRight,
  Plus,
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Card } from '@/components/ui/card';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence?: number;
}

interface ImageData {
  name: string;
  data: string;
  width: number;
  height: number;
}

interface Props {
  imageName: string;
}

export default function ImageAnnotator({ imageName }: Props) {
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [boxes, setBoxes] = useState<BoundingBox[]>([]);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [selectedBox, setSelectedBox] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<BoundingBox | null>(null);
  const [labelInput, setLabelInput] = useState('');
  const [labelContext, setLabelContext] = useState('');
  const [predictLabels, setPredictLabels] = useState('');
  const [loading, setLoading] = useState(false);
  const [scale, setScale] = useState(1);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeHandle, setResizeHandle] = useState<string | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);
  const [editingBox, setEditingBox] = useState<BoundingBox | null>(null);
  const [dinoThreshold, setDinoThreshold] = useState(0.15); // DINO confidence threshold
  const [tool, setTool] = useState<'select' | 'hand'>('select');
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number } | null>(null);
  const [showClassLegend, setShowClassLegend] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadImage();
  }, [imageName]);

  // Persist state across tab switches
  useEffect(() => {
    const savedState = localStorage.getItem(`annotator-${imageName}`);
    if (savedState) {
      try {
        const { boxes: savedBoxes, classNames: savedClasses, scale: savedScale } = JSON.parse(savedState);
        if (savedBoxes) setBoxes(savedBoxes);
        if (savedClasses) setClassNames(savedClasses);
        if (savedScale) setScale(savedScale);
      } catch (e) {
        console.error('Error loading saved state:', e);
      }
    }
  }, [imageName]);

  // Save state when boxes or scale changes
  useEffect(() => {
    if (boxes.length > 0 || classNames.length > 0) {
      localStorage.setItem(`annotator-${imageName}`, JSON.stringify({ boxes, classNames, scale }));
    }
  }, [boxes, classNames, scale, imageName]);

  useEffect(() => {
    if (imageData) {
      drawCanvas();
    }
  }, [boxes, selectedBox, currentBox, imageData, scale, isEditMode, editingBox]);

  useEffect(() => {
    // Keyboard shortcuts
    const handleKeyDown = (e: KeyboardEvent) => {
      // Enter to commit editing box and label it
      if (e.key === 'Enter' && isEditMode && editingBox) {
        e.preventDefault();
        commitEditingBox();
      }
      // Delete selected box with Delete or Backspace
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBox !== null && !isEditMode) {
        e.preventDefault();
        handleDeleteBox();
      }
      // Escape to cancel
      if (e.key === 'Escape') {
        if (isEditMode) {
          setIsEditMode(false);
          setEditingBox(null);
        }
        setSelectedBox(null);
        setIsDrawing(false);
        setCurrentBox(null);
      }
      // V key for select tool
      if (e.key === 'v' || e.key === 'V') {
        setTool('select');
      }
      // H key for hand tool
      if (e.key === 'h' || e.key === 'H') {
        setTool('hand');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedBox, boxes, isEditMode, editingBox]);

  const loadImage = async () => {
    try {
      setLoading(true);
      // Fetch image
      const imageResponse = await fetch(`http://localhost:8000/image/${imageName}`);
      const imageData = await imageResponse.json();
      setImageData(imageData);

      // Fetch existing annotations
      const annotationsResponse = await fetch(`http://localhost:8000/annotations/${imageName}`);
      const annotationsData = await annotationsResponse.json();
      setBoxes(annotationsData.boxes || []);
      setClassNames(annotationsData.class_names || []);
    } catch (error) {
      console.error('Error loading image:', error);
    } finally {
      setLoading(false);
    }
  };

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !imageData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas size
      canvas.width = imageData.width;
      canvas.height = imageData.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw boxes
      boxes.forEach((box, index) => {
        const isSelected = index === selectedBox;
        ctx.strokeStyle = isSelected ? '#ff0000' : getColorForLabel(box.label);
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // Draw label
        ctx.fillStyle = isSelected ? '#ff0000' : getColorForLabel(box.label);
        ctx.font = '14px Arial';
        const labelText = box.confidence
          ? `${box.label} (${(box.confidence * 100).toFixed(1)}%)`
          : box.label;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(box.x, box.y - 20, textWidth + 8, 20);
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, box.x + 4, box.y - 5);
      });

      // Draw current box being drawn
      if (currentBox) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(currentBox.x, currentBox.y, currentBox.width, currentBox.height);
        ctx.setLineDash([]);
      }

      // Draw editing box with resize handles (Photoshop style)
      if (isEditMode && editingBox) {
        ctx.strokeStyle = '#ffa500';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(editingBox.x, editingBox.y, editingBox.width, editingBox.height);
        ctx.setLineDash([]);

        // Draw resize handles (larger, Photoshop style)
        const handleSize = 10;
        const borderSize = 2;
        const handles = [
          { x: editingBox.x, y: editingBox.y }, // top-left
          { x: editingBox.x + editingBox.width, y: editingBox.y }, // top-right
          { x: editingBox.x, y: editingBox.y + editingBox.height }, // bottom-left
          { x: editingBox.x + editingBox.width, y: editingBox.y + editingBox.height }, // bottom-right
          // Mid handles
          { x: editingBox.x + editingBox.width / 2, y: editingBox.y }, // top-mid
          { x: editingBox.x + editingBox.width / 2, y: editingBox.y + editingBox.height }, // bottom-mid
          { x: editingBox.x, y: editingBox.y + editingBox.height / 2 }, // left-mid
          { x: editingBox.x + editingBox.width, y: editingBox.y + editingBox.height / 2 }, // right-mid
        ];

        handles.forEach(handle => {
          // Draw white fill
          ctx.fillStyle = 'white';
          ctx.fillRect(handle.x - handleSize/2, handle.y - handleSize/2, handleSize, handleSize);
          // Draw orange border
          ctx.strokeStyle = '#ffa500';
          ctx.lineWidth = borderSize;
          ctx.strokeRect(handle.x - handleSize/2, handle.y - handleSize/2, handleSize, handleSize);
        });

        // Draw "Press Enter to label" text
        ctx.fillStyle = 'rgba(255, 165, 0, 0.9)';
        ctx.font = 'bold 14px Arial';
        const instructionText = 'Press Enter to commit';
        const textW = ctx.measureText(instructionText).width;
        ctx.fillRect(editingBox.x, editingBox.y + editingBox.height + 5, textW + 10, 22);
        ctx.fillStyle = 'white';
        ctx.fillText(instructionText, editingBox.x + 5, editingBox.y + editingBox.height + 20);
      }

    };
    img.src = imageData.data;
  };

  const getColorForLabel = (label: string): string => {
    const colors = [
      '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
      '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
    ];
    const index = classNames.indexOf(label);
    return colors[index % colors.length];
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Hand tool - enable panning
    if (tool === 'hand') {
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    // If in edit mode, check for handle clicks first
    if (isEditMode && editingBox) {
      const handleSize = 8;
      const handles = [
        { name: 'tl', x: editingBox.x, y: editingBox.y },
        { name: 'tr', x: editingBox.x + editingBox.width, y: editingBox.y },
        { name: 'bl', x: editingBox.x, y: editingBox.y + editingBox.height },
        { name: 'br', x: editingBox.x + editingBox.width, y: editingBox.y + editingBox.height },
      ];

      for (const handle of handles) {
        if (Math.abs(x - handle.x) < handleSize && Math.abs(y - handle.y) < handleSize) {
          setIsResizing(true);
          setResizeHandle(handle.name);
          return;
        }
      }

      // Check if clicking inside editing box to move it
      if (x >= editingBox.x && x <= editingBox.x + editingBox.width &&
          y >= editingBox.y && y <= editingBox.y + editingBox.height) {
        setIsResizing(true);
        setResizeHandle('move');
        setDrawStart({ x, y });
        return;
      }
    }

    // Check if clicking on existing box
    const clickedBoxIndex = boxes.findIndex(box =>
      x >= box.x && x <= box.x + box.width &&
      y >= box.y && y <= box.y + box.height
    );

    if (clickedBoxIndex !== -1) {
      setSelectedBox(clickedBoxIndex);
    } else {
      setSelectedBox(null);
      setIsDrawing(true);
      setDrawStart({ x, y });
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Handle panning
    if (isPanning && panStart && containerRef.current) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      containerRef.current.scrollLeft -= dx;
      containerRef.current.scrollTop -= dy;
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    // Handle resizing editing box
    if (isResizing && editingBox && resizeHandle) {
      const newBox = { ...editingBox };

      if (resizeHandle === 'move' && drawStart) {
        const dx = x - drawStart.x;
        const dy = y - drawStart.y;
        newBox.x += dx;
        newBox.y += dy;
        setDrawStart({ x, y });
      } else if (resizeHandle === 'tl') {
        const dx = x - newBox.x;
        const dy = y - newBox.y;
        newBox.x = x;
        newBox.y = y;
        newBox.width -= dx;
        newBox.height -= dy;
      } else if (resizeHandle === 'tr') {
        const dy = y - newBox.y;
        newBox.y = y;
        newBox.width = x - newBox.x;
        newBox.height -= dy;
      } else if (resizeHandle === 'bl') {
        const dx = x - newBox.x;
        newBox.x = x;
        newBox.width -= dx;
        newBox.height = y - newBox.y;
      } else if (resizeHandle === 'br') {
        newBox.width = x - newBox.x;
        newBox.height = y - newBox.y;
      }

      setEditingBox(newBox);
      return;
    }

    // Handle drawing new box
    if (isDrawing && drawStart) {
      const width = x - drawStart.x;
      const height = y - drawStart.y;

      setCurrentBox({
        x: width > 0 ? drawStart.x : x,
        y: height > 0 ? drawStart.y : y,
        width: Math.abs(width),
        height: Math.abs(height),
        label: ''
      });
    }
  };

  const handleCanvasMouseUp = () => {
    // End panning
    if (isPanning) {
      setIsPanning(false);
      setPanStart(null);
      return;
    }

    // End resizing
    if (isResizing) {
      setIsResizing(false);
      setResizeHandle(null);
      setDrawStart(null);
      return;
    }

    // Prevent double execution
    if (!isDrawing || !currentBox) return;

    if (currentBox.width > 10 && currentBox.height > 10) {
      // Enter EDIT MODE - let user adjust the box
      setEditingBox({ ...currentBox });
      setIsEditMode(true);
      setIsDrawing(false);
      setDrawStart(null);
      setCurrentBox(null);
    } else {
      // Box too small, just clear state
      setIsDrawing(false);
      setDrawStart(null);
      setCurrentBox(null);
    }
  };

  const commitEditingBox = async () => {
    if (!editingBox) return;

    const label = labelInput.trim();
    if (!label) {
      alert('Please set a label in the toolbar before confirming the box.');
      return;
    }

    const context = labelContext.trim();

    // Add the manually drawn box first so the user always keeps their reference
    const newBox = { ...editingBox, label };
    setBoxes((prev) => [...prev, newBox]);
    setClassNames((prev) => (prev.includes(label) ? prev : [...prev, label]));

    // Clear edit mode
    setIsEditMode(false);
    setEditingBox(null);

    // Automatically detect similar objects using VISUAL SIMILARITY
    setLoading(true);

    try {
      // Use the visual similarity endpoint powered by Qwen3-VL
      const response = await fetch('http://localhost:8000/predict_visual_similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: imageName,
          crop_box: {
            x: editingBox.x,
            y: editingBox.y,
            width: editingBox.width,
            height: editingBox.height,
            label: label
          },
          similarity_threshold: dinoThreshold,
          dino_threshold: dinoThreshold,
          label_context: context || null
        })
      });

      const data = await response.json();

      console.log('Response from backend:', data);
      console.log('Number of boxes received:', data.boxes?.length || 0);

      if (data.boxes && data.boxes.length > 0) {
        // Remove duplicates (boxes that are too close to each other)
        const uniqueBoxes: BoundingBox[] = [];
        data.boxes.forEach((box: BoundingBox & { similarity: number }) => {
          console.log('Processing box:', box);
          const isDuplicate = uniqueBoxes.some(existing =>
            Math.abs(existing.x - box.x) < 30 && Math.abs(existing.y - box.y) < 30
          );
          if (!isDuplicate) {
            uniqueBoxes.push({ ...box, label });
          }
        });

        console.log('Unique boxes after filtering:', uniqueBoxes.length);

        // Add all similar boxes with the user's label
        if (uniqueBoxes.length > 0) {
          setBoxes(prev => {
            const newBoxes = [...prev, ...uniqueBoxes];
            console.log('Total boxes after adding:', newBoxes.length);
            return newBoxes;
          });
          console.log(`Auto-detected ${uniqueBoxes.length} visually similar objects for "${label}"`);
        } else {
          console.log('No similar objects found. Try lowering the similarity threshold.');
        }
      } else {
        console.log('No boxes in response or empty array');
      }
    } catch (error) {
      console.error('Error auto-detecting:', error);
      alert('Error detecting similar objects. Make sure the backend is running with Qwen3-VL model.');
    } finally {
      setLoading(false);
    }
  };

  const handleCanvasMouseLeave = () => {
    // Just cancel drawing, don't trigger label prompt
    setIsDrawing(false);
    setDrawStart(null);
    setCurrentBox(null);
  };

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    // Zoom with mouse wheel (Ctrl + scroll)
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setScale(Math.max(0.25, Math.min(4, scale + delta)));
    }
  };

  const handlePredict = async () => {
    if (!predictLabels.trim()) return;

    try {
      setLoading(true);
      const labels = predictLabels.split(',').map(l => l.trim()).filter(l => l);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: imageName,
          text_labels: labels,
          threshold: 0.4,
          text_threshold: 0.3
        })
      });

      const data = await response.json();

      // Check if response has boxes array
      if (!data.boxes || !Array.isArray(data.boxes)) {
        console.error('Invalid response from backend:', data);
        alert('Error: Backend returned invalid data. Check console for details.');
        return;
      }

      if (data.boxes.length === 0) {
        alert('No objects detected. Try lowering the threshold or using different labels.');
        return;
      }

      // Add new boxes
      setBoxes([...boxes, ...data.boxes]);

      // Update class names
      const newClasses = new Set([...classNames]);
      data.boxes.forEach((box: BoundingBox) => newClasses.add(box.label));
      setClassNames(Array.from(newClasses));
    } catch (error) {
      console.error('Error predicting:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setLoading(true);
      await fetch('http://localhost:8000/save_annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: imageName,
          boxes,
          class_names: classNames
        })
      });
      alert('Annotations saved successfully!');
    } catch (error) {
      console.error('Error saving:', error);
      alert('Error saving annotations');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteBox = () => {
    if (selectedBox !== null) {
      setBoxes(boxes.filter((_, i) => i !== selectedBox));
      setSelectedBox(null);
    }
  };

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all annotations?')) {
      setBoxes([]);
      setSelectedBox(null);
    }
  };

  if (loading && !imageData) {
    return (
      <div className="flex items-center justify-center h-full bg-[#0a0a0a]">
        <div className="text-xl text-gray-300">Loading...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      {/* Modern Toolbar with better contrast */}
      <div className="bg-zinc-900 border-b border-zinc-800 p-3 shadow-lg">
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="font-semibold text-sm text-zinc-50 px-2">{imageName}</h2>
          <div className="h-8 w-px bg-zinc-700" />

          {/* Tool Switcher - Better contrast */}
          <div className="flex items-center gap-1 bg-zinc-800/50 rounded-md p-1">
            <Button
              variant={tool === 'select' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setTool('select')}
              className={`gap-1.5 ${tool === 'select' ? 'bg-blue-600 hover:bg-blue-700' : 'hover:bg-zinc-700'}`}
            >
              <MousePointer2 className="w-4 h-4" />
              Select
            </Button>
            <Button
              variant={tool === 'hand' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setTool('hand')}
              className={`gap-1.5 ${tool === 'hand' ? 'bg-blue-600 hover:bg-blue-700' : 'hover:bg-zinc-700'}`}
            >
              <Hand className="w-4 h-4" />
              Pan
            </Button>
          </div>

          <div className="h-8 w-px bg-zinc-700" />

          {/* Zoom Controls - Better visibility */}
          <div className="flex items-center gap-2 bg-zinc-800/30 rounded-md px-2 py-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 hover:bg-zinc-700 hover:text-zinc-50"
              onClick={() => setScale(Math.max(0.25, scale - 0.25))}
            >
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Slider
              value={[scale]}
              onValueChange={([v]) => setScale(v)}
              min={0.25}
              max={4}
              step={0.25}
              className="w-28"
            />
            <Badge variant="secondary" className="w-16 justify-center font-semibold bg-zinc-800 text-zinc-100 border-zinc-700">
              {Math.round(scale * 100)}%
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 hover:bg-zinc-700 hover:text-zinc-50"
              onClick={() => setScale(Math.min(4, scale + 0.25))}
            >
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 hover:bg-zinc-700 hover:text-zinc-50"
              onClick={() => setScale(1)}
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>

          <div className="h-8 w-px bg-zinc-700" />

          {/* DINO Threshold - Better contrast */}
          <div className="flex items-center gap-2 bg-zinc-800/30 rounded-md px-3 py-1">
            <span className="text-xs text-zinc-300 font-medium uppercase tracking-wide">DINO</span>
            <Slider
              value={[dinoThreshold]}
              onValueChange={([v]) => setDinoThreshold(v)}
              min={0.15}
              max={0.40}
              step={0.05}
              className="w-28"
            />
            <Badge className="w-16 justify-center font-mono bg-zinc-800 text-zinc-100 border border-zinc-700 hover:bg-zinc-700">
              {(dinoThreshold * 100).toFixed(0)}%
            </Badge>
          </div>

          <div className="h-8 w-px bg-zinc-700" />

          {/* AI Model Badge - More prominent */}
          <Badge className="bg-purple-600 hover:bg-purple-700 text-white px-3 py-1.5 gap-1.5 shadow-lg shadow-purple-900/50">
            <Sparkles className="w-3.5 h-3.5" />
            Qwen3-VL
          </Badge>

          <div className="flex-1" />

          <Button
            onClick={handleSave}
            disabled={loading}
            size="sm"
            className="gap-2 bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save
          </Button>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-[220px_minmax(0,1fr)]">
          <div className="flex flex-col gap-1 bg-zinc-800/30 rounded-md px-3 py-2">
            <label className="text-xs text-zinc-300 font-medium uppercase tracking-wide" htmlFor="label-input">
              Label
            </label>
            <input
              id="label-input"
              type="text"
              value={labelInput}
              onChange={(e) => setLabelInput(e.target.value)}
              placeholder="e.g. pepsodent_powder"
              className="px-2 py-1.5 rounded bg-[#1a1a1a] border border-zinc-700 text-sm text-zinc-50 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div className="flex flex-col gap-1 bg-zinc-800/30 rounded-md px-3 py-2">
            <div className="flex items-center justify-between">
              <label className="text-xs text-zinc-300 font-medium uppercase tracking-wide" htmlFor="context-input">
                Context
              </label>
              <span className="text-[10px] text-zinc-500 uppercase">Optional</span>
            </div>
            <input
              id="context-input"
              type="text"
              value={labelContext}
              onChange={(e) => setLabelContext(e.target.value)}
              placeholder="Describe packaging cues, variant, colors"
              className="px-2 py-1.5 rounded bg-[#1a1a1a] border border-zinc-700 text-sm text-zinc-50 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Manual text-based detection (optional - use drawing for better accuracy) */}
        <div className="flex items-center gap-2 opacity-75">
          <input
            type="text"
            value={predictLabels}
            onChange={(e) => setPredictLabels(e.target.value)}
            placeholder="Optional: Text-based detection (less accurate - use drawing instead)"
            className="flex-1 px-3 py-2 border border-[#3a3a3a] rounded bg-[#1a1a1a] text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handlePredict}
            disabled={loading || !predictLabels.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:bg-gray-700 disabled:text-gray-500 transition-colors"
            title="Text-based detection (Grounding DINO only - no visual similarity)"
          >
            <Sparkles className="w-4 h-4" />
            <span>Text Detect</span>
          </button>
        </div>

        {/* Box controls */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{boxes.length} annotations</span>
          <button
            onClick={handleDeleteBox}
            disabled={selectedBox === null}
            className="flex items-center gap-1.5 px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Delete Selected</span>
          </button>
          <button
            onClick={handleClearAll}
            disabled={boxes.length === 0}
            className="flex items-center gap-1.5 px-3 py-1 text-sm bg-orange-600 text-white rounded hover:bg-orange-700 disabled:bg-gray-700 disabled:text-gray-500 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear All</span>
          </button>
        </div>

        {/* Class legend - Collapsible */}
        {classNames.length > 0 && (
          <div className="border-t border-[#3a3a3a] pt-3">
            <button
              onClick={() => setShowClassLegend(!showClassLegend)}
              className="flex items-center gap-2 px-3 py-2 bg-[#2a2a2a] text-white rounded hover:bg-[#3a3a3a] transition-colors text-sm w-full justify-between"
            >
              <div className="flex items-center gap-2">
                {showClassLegend ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                <span>Classes ({classNames.length})</span>
              </div>
              <span className="text-xs text-gray-400">Click to {showClassLegend ? 'hide' : 'show'} • Click chips to quick-add</span>
            </button>
            {showClassLegend && (
              <div className="flex flex-wrap gap-2 mt-2 p-2 bg-[#151515] rounded">
                {classNames.map((className) => (
                  <button
                    key={className}
                    onClick={() => {
                      setPredictLabels(className);
                      handlePredict();
                    }}
                    className="flex items-center gap-2 px-3 py-1 rounded text-sm text-white shadow-md hover:opacity-80 transition-opacity cursor-pointer"
                    style={{ backgroundColor: getColorForLabel(className) }}
                    title={`Click to auto-detect more "${className}" objects`}
                  >
                    {className}
                    <span className="text-xs opacity-75">
                      ({boxes.filter(b => b.label === className).length})
                    </span>
                    <Plus className="w-3 h-3 font-bold" />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Canvas area */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto bg-zinc-950 p-8"
        onWheel={handleWheel}
      >
        <div className="inline-block transition-transform duration-100 ease-out" style={{ transform: `scale(${scale})`, transformOrigin: 'top left' }}>
          <canvas
            ref={canvasRef}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseLeave}
            className={`border border-zinc-800 shadow-2xl shadow-black/50 rounded-lg transition-shadow hover:shadow-zinc-900/50 ${
              tool === 'hand' ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-crosshair'
            }`}
            style={{ display: 'block' }}
          />
        </div>
      </div>

      {/* Modern Instructions Footer */}
      <div className="bg-zinc-900/50 backdrop-blur-sm border-t border-zinc-800 px-4 py-3">
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="gap-1">
              <MousePointer2 className="w-3 h-3" />
              Draw
            </Badge>
            <span className="text-zinc-500">→</span>
            <Badge variant="outline">Enter</Badge>
            <span className="text-zinc-500">→</span>
            <Badge variant="outline" className="gap-1">
              <Sparkles className="w-3 h-3" />
              AI Detects
            </Badge>
          </div>
          <Separator orientation="vertical" className="h-4 bg-zinc-700" />
          <div className="flex gap-3 text-xs text-zinc-400">
            <kbd className="px-1.5 py-0.5 bg-zinc-800 rounded border border-zinc-700">V</kbd><span>Select</span>
            <kbd className="px-1.5 py-0.5 bg-zinc-800 rounded border border-zinc-700">H</kbd><span>Pan</span>
            <kbd className="px-1.5 py-0.5 bg-zinc-800 rounded border border-zinc-700">Ctrl+Scroll</kbd><span>Zoom</span>
            <kbd className="px-1.5 py-0.5 bg-zinc-800 rounded border border-zinc-700">Del</kbd><span>Remove</span>
          </div>
        </div>
      </div>
    </div>
  );
}
