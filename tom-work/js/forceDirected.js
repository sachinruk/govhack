
var updateLinks = function() {
    $.ajax({
        url: 'data/adj_mat/adj_mat1.csv'
    }).done(function(data) {
        var dataArray = $.csv.toArrays(data);

        var newLinksArray = [];
        for(var i=1; i<dataArray.length; i++) {
            for(var j=1; j<dataArray[i].length; j++) {
                if(dataArray[i][j] == 1) {
                    var newLink = {};
                    newLink.source = i-1;
                    newLink.target = j-1;

                    newLinksArray.push(newLink);
                }
            }
        }

        svg.selectAll(".link").remove();
        svg.selectAll(".node").remove();

        force.links(newLinksArray).start();

        link = svg.selectAll(".link")
            .data(newLinksArray)
            .enter().append("line")
            .attr("class", "link")
            .style("stroke-width", strokeWidth);

        node = svg.selectAll(".node")
            .data(graph.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(force.drag)
            .on('click', connectedNodes); //Added code 
        node.append("circle")
            .attr("r", 8)
            .style("fill", nodeColor)
        node.append("text")
              .attr("dx", 10)
              .attr("dy", ".35em")
              .text(function(d) { return d.name })
              .style("stroke", "gray");

        linkedByIndex = {};
        for (i = 0; i < newLinksArray.length; i++) {
            linkedByIndex[i + "," + i] = 1;
        };
        newLinksArray.forEach(function (d) {
            linkedByIndex[d.source.index + "," + d.target.index] = 1;
        });

    }).fail(function() {
        alert('Could not retrive data');
    });
}


//Constants for the SVG
var width = 1000,
    height = 600,
    strokeWidth = 1,
    nodeColor = '#6c217f';

//Set up the colour scale
var color = d3.scale.category20();

//Set up the force layout
var force = d3.layout.force()
    .charge(-420)
    .linkDistance(150)
    .size([width, height]);

//Append a SVG to the body of the html page. Assign this SVG as an object to svg
var svg = d3.select(".forceDirectedGraph").append("svg")
    .attr("width", width)
    .attr("height", height);

//Read the data from the mis element 
var mis = document.getElementById('mis').innerHTML;
graph = JSON.parse(mis);

//Creates the graph data structure out of the json data
force.nodes(graph.nodes)
    .links(graph.links)
    .start();

//Create all the line svgs but without locations yet
var link = svg.selectAll(".link")
    .data(graph.links)
    .enter().append("line")
    .attr("class", "link")
    .style("stroke-width", strokeWidth);

var node = svg.selectAll(".node")
    .data(graph.nodes)
    .enter().append("g")
    .attr("class", "node")
    .call(force.drag)
    .on('dblclick', connectedNodes); //Added code 
node.append("circle")
    .attr("r", 8)
    .style("fill", nodeColor)
node.append("text")
      .attr("dx", 10)
      .attr("dy", ".35em")
      .text(function(d) { return d.name })
      .style("stroke", "gray");


updateLinks();



//Now we are giving the SVGs co-ordinates - the force layout is generating the co-ordinates which this code is using to update the attributes of the SVG elements
force.on("tick", function () {
    link.attr("x1", function (d) {
        return d.source.x;
    })
        .attr("y1", function (d) {
        return d.source.y;
    })
        .attr("x2", function (d) {
        return d.target.x;
    })
        .attr("y2", function (d) {
        return d.target.y;
    });
    d3.selectAll("circle").attr("cx", function (d) {
        return d.x;
    })
        .attr("cy", function (d) {
        return d.y;
    });
    d3.selectAll("text").attr("x", function (d) {
        return d.x;
    })
        .attr("y", function (d) {
        return d.y;
    });
});


//Toggle stores whether the highlighting is on
var toggle = 0;
//Create an array logging what is connected to what
var linkedByIndex = {};
for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
};
graph.links.forEach(function (d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
});

//This function looks up whether a pair are neighbours
function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
}

function neighboringByIndex(a, b) {
    return linkedByIndex[a + ',' + b];
}

function connectedNodes() {
    d = d3.select(this).node().__data__;

    doOpacity(d);

    LineGraph.removeLines();

    // Show related cancers in a different color NOT WORKING
    // for(var currentIndex=0; currentIndex<Object.keys(linkedByIndex).length; currentIndex++) {
    //     if(currentIndex == d.index) {
    //         ; 
    //     }          
    //     else if(neighboringByIndex(d.index, currentIndex)) {
    //         var cancerName = d3.select('.node:nth-child(' + currentIndex + ')');
    //     }
    // }

    var name = d.name;
    $('.currentSelection').text(name);
    name = name.replace(/ /g, '_');
    //LineGraph.setupLineGraph('data/deaths_' + d.index + '.csv', ' primary');
    LineGraph.setupLineGraph('data/cancer_tsvs/' + name + '__FI.tsv', ' primary');
}

function doOpacity(d) {
    if (toggle == 0) {
        //Reduce the opacity of all but the neighbouring nodes
        node.style("opacity", function (o) {
            return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
        });
        link.style("opacity", function (o) {
            return d.index==o.source.index | d.index==o.target.index ? 1 : 0.1;
        });
        //Reduce the op
        toggle = 1;
    } else {
        //Put them back to opacity=1
        node.style("opacity", 1);
        link.style("opacity", 1);
        toggle = 0;
    }
}

// NOT WORKING
function selectNode(nodeName) {
    for(var index=0; index<graph.nodes.length; index++) {
        if(graph.nodes[index].name == nodeName) {
            var node = d3.select('.node:nth-child(' + index + ')').node().__data__;
            console.log(node);
            doOpacity(node);
        }
    }
}