import React from 'react';
import { TabType } from '../App';

interface SidebarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  tabs: {
    id: TabType;
    label: string;
    icon: React.ReactNode;
  }[];
}

const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, tabs }) => {
  return (
    <aside className="w-64 border-r bg-gray-50 dark:bg-gray-800 dark:border-gray-700 flex flex-col">
      <nav className="flex-1 py-4">
        <ul>
          {tabs.map((tab) => (
            <li key={tab.id} className="mb-1">
              <button
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center px-4 py-3 rounded-lg transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                    : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                }`}
              >
                <span className="mr-3">{tab.icon}</span>
                <span className="font-medium">{tab.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <div className="p-4 border-t dark:border-gray-700">
        <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg">
          <h3 className="font-medium text-blue-700 dark:text-blue-300 mb-1">Info</h3>
          <p className="text-sm text-blue-600 dark:text-blue-400">
            DRL-MOA model for solving multi-objective TSP problems based on deep reinforcement learning.
          </p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;