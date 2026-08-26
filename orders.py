import json
import os

class OrderLookupSystem:
    def __init__(self, orders_file='data/orders.json'):
        self.orders_file = orders_file
        self.orders = {}
        self.load_orders()
        
    def load_orders(self):
        """Loads orders from the JSON file and index them by order_id."""
        if not os.path.exists(self.orders_file):
            print(f"Error: {self.orders_file} not found.")
            return
            
        with open(self.orders_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # We index the orders in a dictionary for fast O(1) lookup
        for order in data.get('orders', []):
            order_id = order.get('order_id', '').strip().upper()
            self.orders[order_id] = order
            
    def lookup_order(self, order_id):
        """
        Looks up an order by ID, applies safety rules, sanitizes data, 
        and handles stale properties.
        
        Returns:
            dict: Sanitized, customer-safe order details or an error message.
        """
        # 1. Normalize ID (Trim spaces and uppercase)
        if not order_id:
            return {"error": "Missing order ID."}
            
        normalized_id = str(order_id).strip().upper()
        
        # 2. Check if order exists
        if normalized_id not in self.orders:
            return {"error": f"Order {normalized_id} not found."}
            
        raw_order = self.orders[normalized_id]
        
        # 3. Create a sanitized, customer-safe version
        # ONLY copy fields that are explicitly allowed in the data dictionary
        sanitized_order = {
            "order_id": raw_order.get("order_id"),
            "membership_tier": raw_order.get("membership_tier"),
            "placed_at": raw_order.get("placed_at"),
            "status": raw_order.get("status"),
            "status_updated_at": raw_order.get("status_updated_at"),
            "customer_safe_message": raw_order.get("customer_safe_message")
        }
        
        # Format items safely (only quantity, name, final_sale)
        items = []
        for item in raw_order.get("items", []):
            items.append({
                "name": item.get("name"),
                "quantity": item.get("quantity"),
                "final_sale": item.get("final_sale")
            })
        sanitized_order["items"] = items
        
        # 4. Apply Business Rules for status-based data visibility
        status = raw_order.get("status", "").lower()
        
        if status in ['cancelled', 'returned']:
            # Do NOT report carrier, tracking, or estimated delivery if cancelled/returned
            sanitized_order["carrier"] = None
            sanitized_order["tracking_number"] = None
            sanitized_order["estimated_delivery"] = None
            sanitized_order["warning"] = f"This order has been {status}. Any estimated delivery dates are stale and should be ignored."
        else:
            # For active statuses, copy standard delivery details
            sanitized_order["carrier"] = raw_order.get("carrier")
            sanitized_order["tracking_number"] = raw_order.get("tracking_number")
            sanitized_order["estimated_delivery"] = raw_order.get("estimated_delivery")
            
        # 5. Handle special 'exception' status
        if status == 'exception':
            sanitized_order["handoff_required"] = True
            sanitized_order["warning"] = "This shipment has encountered an exception and requires human support review."
            
        return sanitized_order

# Simple test code to verify orders system works
if __name__ == '__main__':
    system = OrderLookupSystem()
    
    # Test normalization and valid lookup
    print("\n--- Test Valid Lookup (ORD-1007) ---")
    print(json.dumps(system.lookup_order("  ord-1007  "), indent=2))
    
    # Test cancelled order (should not have carrier/tracking/ETA details)
    print("\n--- Test Cancelled Order (ORD-1004) ---")
    print(json.dumps(system.lookup_order("ORD-1004"), indent=2))
    
    # Test exception order (should flag handoff)
    print("\n--- Test Exception Order (ORD-1010) ---")
    print(json.dumps(system.lookup_order("ORD-1010"), indent=2))
    
    # Test non-existent order
    print("\n--- Test Missing Order (ORD-9999) ---")
    print(json.dumps(system.lookup_order("ORD-9999"), indent=2))
