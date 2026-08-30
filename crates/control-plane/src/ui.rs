//! Minimal UI for dashboard — consumes `TenantDashboard` `dashboard.rs:13`.

pub fn render_html(dash: &[crate::dashboard::TenantDashboard]) -> String {
    let mut html = String::from("<html><body><h1>Traverse</h1>");
    for tenant in dash {
        html.push_str(&format!(
            "<h2>{} pulls={}</h2><ul>",
            tenant.tenant, tenant.total_pulls
        ));
        for arm in &tenant.arms {
            html.push_str(&format!(
                "<li>{}: mean {:.3} pulls {}</li>",
                arm.id, arm.mean, arm.pulls
            ));
        }
        html.push_str("</ul>");
    }
    html.push_str("</body></html>");
    html
}
