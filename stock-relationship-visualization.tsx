import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import Papa from 'papaparse';

const StockNetworkVisualization = () => {
  const svgRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [sectors, setSectors] = useState([]);
  const [selectedView, setSelectedView] = useState('sectors');

  useEffect(() => {
    async function loadData() {
      try {
        // Load stock_relationships data
        const relationsResponse = await window.fs.readFile('stock_relationships.csv');
        const relationsText = new TextDecoder().decode(relationsResponse);
        const relationsParsed = Papa.parse(relationsText, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true
        });

        createVisualization(relationsParsed.data);
        setLoading(false);
      } catch (error) {
        console.error("Error loading data:", error);
      }
    }

    loadData();
  }, [selectedView]);

  const createVisualization = (relationships) => {
    if (!svgRef.current) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();

    // Filter relationships based on selected view
    const filteredRelations = selectedView === 'sectors' 
      ? relationships.filter(r => r.relation_type === 'sector')
      : selectedView === 'correlated' 
        ? relationships.filter(r => r.relation_type === 'correlated')
        : relationships;

    // Extract nodes (unique stocks)
    const nodes = Array.from(new Set([
      ...filteredRelations.map(r => r.stock1),
      ...filteredRelations.map(r => r.stock2)
    ])).map(id => ({ id }));

    // Create links with source and target indices
    const links = filteredRelations.map(r => ({
      source: nodes.findIndex(n => n.id === r.stock1),
      target: nodes.findIndex(n => n.id === r.stock2),
      value: r.weight,
      type: r.relation_type
    }));

    // Set up SVG dimensions
    const width = 800;
    const height = 600;

    // Create SVG container
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('style', 'max-width: 100%; height: auto;');

    // Set up the force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink().links(links).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Define color scale for relationship types
    const colorScale = d3.scaleOrdinal()
      .domain(['sector', 'correlated'])
      .range(['#4e79a7', '#f28e2c']);

    // Draw links
    const link = svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', d => Math.sqrt(d.value) * 2)
      .attr('stroke', d => colorScale(d.type));

    // Draw nodes
    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', 8)
      .attr('fill', '#69b3a2')
      .call(drag(simulation));

    // Add labels to nodes
    const labels = svg.append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .text(d => d.id)
      .attr('font-size', 10)
      .attr('dx', 12)
      .attr('dy', 4);

    // Update positions on each tick of the simulation
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      labels
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });

    // Extract sector groups for the legend
    if (selectedView === 'sectors') {
      // Find connected components (potential sectors)
      const groupedNodes = {};
      
      // Helper function to find all connected nodes
      const findConnected = (nodeId, group) => {
        if (!groupedNodes[nodeId]) {
          groupedNodes[nodeId] = group;
          
          // Find all nodes connected to this one
          filteredRelations
            .filter(r => r.stock1 === nodeId || r.stock2 === nodeId)
            .forEach(r => {
              const connectedNode = r.stock1 === nodeId ? r.stock2 : r.stock1;
              if (!groupedNodes[connectedNode]) {
                findConnected(connectedNode, group);
              }
            });
        }
      };
      
      // Assign nodes to groups
      let groupCounter = 1;
      nodes.forEach(node => {
        if (!groupedNodes[node.id]) {
          findConnected(node.id, groupCounter++);
        }
      });
      
      // Group nodes by sector
      const sectorGroups = {};
      Object.entries(groupedNodes).forEach(([nodeId, groupId]) => {
        if (!sectorGroups[groupId]) sectorGroups[groupId] = [];
        sectorGroups[groupId].push(nodeId);
      });
      
      // Set sectors for display
      setSectors(Object.entries(sectorGroups)
        .map(([groupId, stocks]) => ({
          id: `Sector ${groupId}`,
          stocks: stocks.join(', ')
        })));
    }
  };

  // Drag functionality for the nodes
  const drag = (simulation) => {
    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }
    
    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }
    
    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
    
    return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  };

  return (
    <div className="flex flex-col items-center w-full p-4">
      <h2 className="text-2xl font-bold mb-4">Stock Relationship Network</h2>
      
      <div className="mb-4 flex space-x-2">
        <button 
          className={`px-4 py-2 rounded ${selectedView === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => setSelectedView('all')}
        >
          All Relationships
        </button>
        <button 
          className={`px-4 py-2 rounded ${selectedView === '