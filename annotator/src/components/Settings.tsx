'use client';

import { useState, useEffect } from 'react';
import { X, Plus, Trash2, Info } from 'lucide-react';

interface SettingsProps {
  onClose: () => void;
}

export default function Settings({ onClose }: SettingsProps) {
  const [config, setConfig] = useState({
    images_path: '',
    labels_path: '',
    dataset_yaml: '',
    classes: [] as string[],
    num_classes: 0
  });
  const [newClassName, setNewClassName] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Error loading config:', error);
    }
  };

  const updatePath = async (field: string, value: string) => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: value })
      });
      const data = await response.json();
      alert(data.message);
      loadConfig();
    } catch (error) {
      console.error('Error updating config:', error);
      alert('Failed to update configuration');
    } finally {
      setLoading(false);
    }
  };

  const addClass = async () => {
    if (!newClassName.trim()) return;

    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'add',
          class_name: newClassName.trim()
        })
      });
      const data = await response.json();
      alert(data.message);
      setNewClassName('');
      loadConfig();
    } catch (error) {
      console.error('Error adding class:', error);
      alert('Failed to add class');
    } finally {
      setLoading(false);
    }
  };

  const deleteClass = async (className: string) => {
    if (!confirm(`Are you sure you want to delete class "${className}"?`)) return;

    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'delete',
          class_name: className
        })
      });
      const data = await response.json();
      alert(data.message);
      loadConfig();
    } catch (error) {
      console.error('Error deleting class:', error);
      alert('Failed to delete class');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-[#1a1a1a] rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-[#2a2a2a]">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">Settings & Configuration</h2>
          <button
            onClick={onClose}
            className="p-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
            title="Close Settings"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Paths Configuration */}
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">Paths Configuration</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Images Path</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={config.images_path}
                    onChange={(e) => setConfig({ ...config, images_path: e.target.value })}
                    className="flex-1 px-3 py-2 bg-[#0a0a0a] border border-[#3a3a3a] rounded text-white"
                  />
                  <button
                    onClick={() => updatePath('images_path', config.images_path)}
                    disabled={loading}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-700"
                  >
                    Update
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Labels Path</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={config.labels_path}
                    onChange={(e) => setConfig({ ...config, labels_path: e.target.value })}
                    className="flex-1 px-3 py-2 bg-[#0a0a0a] border border-[#3a3a3a] rounded text-white"
                  />
                  <button
                    onClick={() => updatePath('labels_path', config.labels_path)}
                    disabled={loading}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-700"
                  >
                    Update
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Dataset YAML Path</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={config.dataset_yaml}
                    onChange={(e) => setConfig({ ...config, dataset_yaml: e.target.value })}
                    className="flex-1 px-3 py-2 bg-[#0a0a0a] border border-[#3a3a3a] rounded text-white"
                  />
                  <button
                    onClick={() => updatePath('dataset_yaml', config.dataset_yaml)}
                    disabled={loading}
                    className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-700"
                  >
                    Update
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Class Management */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">
              Class Management ({config.num_classes} classes)
            </h3>

            {/* Add Class */}
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addClass()}
                placeholder="Enter new class name (e.g., vim_liquid)"
                className="flex-1 px-3 py-2 bg-[#0a0a0a] border border-[#3a3a3a] rounded text-white placeholder-gray-500"
              />
              <button
                onClick={addClass}
                disabled={loading || !newClassName.trim()}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-700 transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Add Class</span>
              </button>
            </div>

            {/* Class List */}
            <div className="bg-[#0a0a0a] border border-[#2a2a2a] rounded p-4 max-h-96 overflow-y-auto">
              {config.classes.length === 0 ? (
                <p className="text-gray-500 text-center py-4">No classes found</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {config.classes.map((className, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between px-3 py-2 bg-[#1a1a1a] rounded border border-[#3a3a3a]"
                    >
                      <span className="text-white text-sm">
                        <span className="text-gray-500 mr-2">{index}:</span>
                        {className}
                      </span>
                      <button
                        onClick={() => deleteClass(className)}
                        disabled={loading}
                        className="flex items-center gap-1 px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 disabled:bg-gray-700 transition-colors"
                      >
                        <Trash2 className="w-3 h-3" />
                        <span>Delete</span>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Info */}
          <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded p-4">
            <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
              <Info className="w-4 h-4" />
              <span>Information</span>
            </h4>
            <ul className="text-sm text-blue-200 space-y-1">
              <li>• Changes to paths require restarting the backend</li>
              <li>• New classes will be added to the dataset.yaml file</li>
              <li>• Deleting a class won't affect existing annotations</li>
              <li>• Class IDs are assigned based on their order in the list</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
