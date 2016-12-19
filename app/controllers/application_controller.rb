class ApplicationController < ActionController::Base
  protect_from_forgery with: :exception

  protected

  def authenticate
    session[:logged_in] or authenticate_or_request_with_http_token do |token|
      token == Rails.application.secrets.authorisation_token
    end
  end
end