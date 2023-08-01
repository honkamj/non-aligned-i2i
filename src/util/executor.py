"""Helper class for executing required nodes of a graph"""

from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple, Union, cast)

Executable = Callable[
    [
        Mapping[str, Any], # inputs
        Mapping[str, Any], # args
        Set[str] # required outputs
    ],
    Optional[Mapping[str, Any]] # outputs
]


class NodeDefinition:
    """Node definition"""
    def __init__(
            self,
            input_names: Iterable[str],
            output_names: Iterable[str],
            executable: Executable
        ) -> None:
        self._input_names = set(input_names)
        self._output_names = set(output_names)
        self._executable = executable

    @property
    def output_names(self) -> Set[str]:
        """Output node names"""
        return self._output_names

    @property
    def input_names(self) -> Set[str]:
        """Input node names"""
        return self._input_names

    @property
    def executable(self) -> Executable:
        """Executable function of the node"""
        return self._executable


class Node:
    """Node"""
    def __init__(
            self,
            inputs: Set['Node'],
            node_definition: NodeDefinition
        ) -> None:
        self._inputs = inputs
        self._node_definition = node_definition

    @property
    def node_definition(self) -> NodeDefinition:
        """Return node definition of the node"""
        return self._node_definition

    @property
    def inputs(self) -> Set['Node']:
        """Input nodes of the node"""
        return self._inputs

    def __repr__(self) -> str:
        return f'<util.executor.Node with executable {self._node_definition.executable.__name__}>'

    def execute(
            self,
            output_pool: Mapping[str, Any],
            required_outputs: Set[str],
            arguments: Mapping[str, Any]
        ) -> Mapping[str, Any]:
        """Execute node"""
        inputs = {
            key: value for key, value in output_pool.items()
            if key in self._node_definition.input_names}
        outputs = self._node_definition.executable(inputs, arguments, required_outputs)
        if outputs is None:
            return {}
        else:
            return outputs


class ExecutionGraph:
    """Execution graph"""
    def __init__(
            self,
            graph_definition: Iterable[NodeDefinition]
        ) -> None:
        self._graph = self._generate_graph(list(graph_definition))

    @classmethod
    def _generate_graph(cls, graph_definition: Sequence[NodeDefinition]) -> List[Node]:
        nodes: Dict[int, Node] = {}
        for node_index in range(len(graph_definition)):
            if node_index not in nodes:
                cls._add_node_to_graph(
                    node_index_to_add=node_index,
                    graph_definition=graph_definition,
                    nodes=nodes
                )
        return list(nodes.values())

    @classmethod
    def _add_node_to_graph(
            cls,
            node_index_to_add: int,
            graph_definition: Sequence[NodeDefinition],
            nodes: Dict[int, Node]
        ) -> None:
        current_node_definition = graph_definition[node_index_to_add]
        input_indices: Set[int] = set()
        for input_name in current_node_definition.input_names:
            for node_index, node_definition in enumerate(graph_definition):
                if input_name in node_definition.output_names:
                    if node_index not in nodes:
                        cls._add_node_to_graph(node_index, graph_definition, nodes)
                    input_indices.add(node_index)
                    break
        nodes[node_index_to_add] = Node(
            inputs = set(
                nodes[input_index] for input_index in input_indices
            ),
            node_definition=current_node_definition
        )

    def get_nodes_to_execute(
            self,
            input_names: Set[str],
            output_names: Set[str]
        ) -> Sequence[Node]:
        """Get nodes to execute"""
        node_pool = set(input_names)
        nodes_to_execute: Set[Node] = self._get_nodes_to_execute_without_order(output_names)
        nodes_to_execute_with_order: List[Node] = []
        while nodes_to_execute:
            nodes_to_remove = set()
            for node in nodes_to_execute:
                if node.node_definition.input_names.issubset(node_pool):
                    nodes_to_execute_with_order.append(node)
                    nodes_to_remove.add(node)
                    node_pool.update(node.node_definition.output_names)
            nodes_to_execute = nodes_to_execute - nodes_to_remove
            if not nodes_to_remove:
                raise RuntimeError(
                    'Could not find possible execution order, '
                    f'check for cyclic graph, remaining nodes: {nodes_to_execute}')
        return nodes_to_execute_with_order

    def _get_nodes_to_execute_without_order(
            self,
            output_names: Set[str]
        ) -> Set[Node]:
        nodes_to_execute: Set[Node] = set()
        for output_name in output_names:
            for node in self._graph:
                if output_name in node.node_definition.output_names:
                    nodes_to_execute.update(
                        self._add_node_to_execution(node)
                    )
                    break
        return nodes_to_execute

    @classmethod
    def _add_node_to_execution(
            cls,
            node: Node
        ) -> Set[Node]:
        nodes: Set[Node] = {node}
        for input_node in node.inputs:
            nodes.update(
                cls._add_node_to_execution(input_node)
            )
        return nodes



class RequiredNodeExecutor:
    """Executor helper for executing only required nodes given outputs"""
    def __init__(
        self,
        graph: ExecutionGraph
    ) -> None:
        self._graph = graph
        self._inputs: List[Set[str]] = []
        self._output_names: List[Set[str]] = []
        self._nodes_to_execute: List[Sequence[Node]] = []

    def execute(
            self,
            inputs: Mapping[str, Any],
            output_names: Sequence[Union[str, None]],
            arguments: Optional[Mapping[str, Any]] = None
        ) -> Tuple[Any, ...]:
        """Execute graph"""
        if arguments is None:
            arguments = {}
        input_names_set = set(inputs.keys())
        output_names_set_with_none = set(output_names)
        output_names_set: Set[str] = cast(
            Set[str],
            output_names_set_with_none - {None}
        )
        output_pool: Dict[str, Any] = dict(inputs)
        nodes_to_execute: Optional[Sequence[Node]] = None
        for cached_inputs, cached_output_names, cached_nodes_to_execute in zip(
                self._inputs,
                self._output_names,
                self._nodes_to_execute):
            if input_names_set == cached_inputs and output_names_set == cached_output_names:
                nodes_to_execute = cached_nodes_to_execute
        if nodes_to_execute is None:
            self._inputs.append(input_names_set)
            self._output_names.append(output_names_set)
            nodes_to_execute = self._graph.get_nodes_to_execute(
                input_names_set,
                output_names_set)
            self._nodes_to_execute.append(nodes_to_execute)
        for node in nodes_to_execute:
            output_pool.update(
                node.execute(
                    output_pool=output_pool,
                    arguments=arguments,
                    required_outputs=output_names_set
                )
            )
        return tuple(
            output_pool[output_name] if output_name is not None else None
            for output_name in output_names
        )
