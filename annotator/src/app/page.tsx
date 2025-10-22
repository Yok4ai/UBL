'use client';

import { useState, useEffect } from 'react';
import ImageAnnotator from '@/components/ImageAnnotator';
import Settings from '@/components/Settings';
import { Settings as SettingsIcon, ChevronLeft, ChevronRight } from 'lucide-react';

export default function Home() {
  const [images, setImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    fetchImages();
  }, []);

  const fetchImages = async () => {
    try {
      const response = await fetch('http://localhost:8000/images');
      const data = await response.json();
      setImages(data.images);
      if (data.images.length > 0) {
        setSelectedImage(data.images[0]);
      }
    } catch (error) {
      console.error('Error fetching images:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#0a0a0a]">
        <div className="text-xl text-gray-300">Loading images...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#0a0a0a]">
      {/* Sidebar */}
      <div
        className={`bg-[#1a1a1a] shadow-2xl overflow-y-auto border-r border-[#2a2a2a] transition-all duration-300 ${
          sidebarCollapsed ? 'w-12' : 'w-64'
        }`}
      >
        {!sidebarCollapsed ? (
          <>
            <div className="p-4 border-b border-[#2a2a2a]">
              <div className="flex justify-between items-start mb-2">
                <h1 className="text-xl font-bold text-white">UBL Annotator</h1>
                <div className="flex gap-1">
                  <button
                    onClick={() => setShowSettings(true)}
                    className="p-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
                    title="Settings & Configuration"
                  >
                    <SettingsIcon className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setSidebarCollapsed(true)}
                    className="p-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
                    title="Collapse Sidebar"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-400">{images.length} images</p>
            </div>
            <div className="p-2">
              {images.map((image) => (
                <button
                  key={image}
                  onClick={() => setSelectedImage(image)}
                  className={`w-full text-left px-3 py-2 rounded mb-1 text-sm truncate transition-colors ${
                    selectedImage === image
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-[#2a2a2a]'
                  }`}
                >
                  {image}
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center py-4 gap-3">
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="p-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
              title="Expand Sidebar"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
              title="Settings"
            >
              <SettingsIcon className="w-4 h-4" />
            </button>
            <div className="text-xs text-gray-400 writing-mode-vertical transform rotate-180 mt-4">
              {images.length} images
            </div>
          </div>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        {selectedImage ? (
          <ImageAnnotator imageName={selectedImage} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500">No image selected</p>
          </div>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  );
}
