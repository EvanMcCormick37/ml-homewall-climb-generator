function HelpPanel() {
  return (
    <div className="help-panel">
      <h4>Controls</h4>
      <ul>
        <li><kbd>Scroll</kbd> to zoom</li>
        <li><kbd>Drag</kbd> to pan (Pan mode)</li>
        <li><kbd>Click</kbd> to add/remove holds</li>
        <li><kbd>F</kbd> to fit image to window</li>
        <li><kbd>1-4</kbd> to switch modes</li>
      </ul>
    </div>
  );
}

export default HelpPanel;
